#!/usr/bin/env python3
"""
dino.py

DINOv3 Patch-similarity script adapted to use the SamDetector class (promptable SAM3 detector)
as the detector source. Minimal changes to original DINO logic — only enough glue to accept
the new detector's (bboxes, centers) outputs.

Expectations:
- Your SAM detector class is named `SamDetector` and exposes a `detect(image, text, threshold, mask_threshold, save_bboxes)`
  method returning (selected_bboxes, centers).
- If SamDetector lives in `detector.py` or `src/detector.py`, the import should succeed.
"""

import math
from functools import lru_cache
from typing import List, Optional, Tuple
import os
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from transformers import AutoModel
from scipy.ndimage import label

# ---------- Attempt to import user's SamDetector ----------
try:
    # Common locations
    from detector import SamDetector
except Exception:
    try:
        from src.detector import SamDetector
    except Exception:
        # If import failed, raise helpful error
        raise ImportError(
            "Could not import SamDetector. Place your detector implementation in detector.py "
            "or src/detector.py and ensure class is named `SamDetector` with a `detect(...)` method."
        )

# ---------- Constants ----------
DEFAULT_TOP_K = 10
NORMALIZE_STATS = {
    # Keep the dataset keys you used previously; update if you have new dataset keys
    'lvd1689m': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'sat493m': {'mean': [0.43, 0.411, 0.296], 'std': [0.213, 0.156, 0.143]},
}


# ---------- DINOv3 patch similarity (kept mostly as-is) ----------
class DINOv3PatchSimilarity:
    def __init__(self, full_model_id: str, device_str: Optional[str] = None):
        self.full_model_id = full_model_id
        self.device_str = device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model_cached(full_model_id, self.device_str)
        self.ps = self.infer_patch_size(self.model, 32)
        self.dataset_key = self.extract_dataset_key(full_model_id)
        self.src_state = None
        self.tgt_state = None

    def extract_dataset_key(self, model_id: str) -> str:
        return model_id.split("-pretrain-")[-1]

    @lru_cache(maxsize=3)
    def load_model_cached(self, full_model_id: str, device_str: str):
        model = AutoModel.from_pretrained(full_model_id).to(device_str)
        model.eval()
        return model

    def infer_patch_size(self, model, default: int):
        if hasattr(model.config, "patch_size"):
            ps = model.config.patch_size
            return ps[0] if isinstance(ps, (tuple, list)) else ps
        return default

    def pad_to_multiple(self, img: Image.Image, multiple: int):
        W, H = img.size
        Wp = int(math.ceil(W / multiple) * multiple)
        Hp = int(math.ceil(H / multiple) * multiple)
        canvas = Image.new("RGB", (Wp, Hp))
        canvas.paste(img, (0, 0))
        return canvas

    def preprocess(self, img: Image.Image):
        img = self.pad_to_multiple(img, self.ps)
        stats = NORMALIZE_STATS.get(self.dataset_key)
        if stats is None:
            raise KeyError(f"Dataset key '{self.dataset_key}' not in NORMALIZE_STATS")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(stats["mean"], stats["std"])
        ])
        return transform(img).unsqueeze(0), np.array(img), img

    def create_patch_state(self, img: Image.Image):
        pv, disp, padded = self.preprocess(img)
        pv = pv.to(self.device_str)
        _, _, H, W = pv.shape
        rows, cols = H // self.ps, W // self.ps

        with torch.no_grad():
            hs = self.model(pixel_values=pv).last_hidden_state[0].cpu().numpy()

        n_patches = rows * cols
        patches = hs[-n_patches:].reshape(rows, cols, -1)
        X = patches.reshape(-1, patches.shape[-1])
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        state = type("PatchState", (), {})()
        state.rows, state.cols = rows, cols
        state.ps = self.ps
        state.X = X
        state.Xn = Xn
        state.disp = disp
        state.padded = padded
        return state

    def process_images(self, src_img: Image.Image, tgt_img: Image.Image):
        self.src_state = self.create_patch_state(src_img)
        self.tgt_state = self.create_patch_state(tgt_img)

    def idx_to_rc(self, idx, cols):
        return idx // cols, idx % cols

    def coords_to_idx(self, x, y, state):
        x = max(0, min(x, state.cols * state.ps - 1))
        y = max(0, min(y, state.rows * state.ps - 1))
        return (y // state.ps) * state.cols + (x // state.ps)

    # ---------- BBOX FROM PATCH INDICES ----------
    def topk_bbox(self, state, indices: List[int]):
        rows, cols, ps = state.rows, state.cols, state.ps
        mask = np.zeros((rows, cols), dtype=bool)

        for idx in indices:
            r, c = self.idx_to_rc(idx, cols)
            mask[r, c] = True

        labels, num = label(mask)
        if num == 0:
            return None

        best_area, best_bbox = 0, None
        for i in range(1, num + 1):
            ys, xs = np.where(labels == i)
            if len(xs) > best_area:
                best_area = len(xs)
                best_bbox = (
                    xs.min() * ps,
                    ys.min() * ps,
                    (xs.max() + 1) * ps,
                    (ys.max() + 1) * ps,
                )
        return best_bbox

    # ---------- PATCH SELECTION WITH THRESHOLD ----------
    def select_patch(
        self,
        x: int,
        y: int,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: Optional[float] = 0.6,
        max_top_k: Optional[int] = None,
        allow_fallback: bool = False,
    ):
        """
        Returns:
            top_patches : List[PIL.Image]
            best_bbox   : Optional[Tuple[int, int, int, int]]
        """

        if self.src_state is None or self.tgt_state is None:
            raise RuntimeError("Source or target state not initialized. Call process_images() first.")

        q_idx = self.coords_to_idx(x, y, self.src_state)
        qn = self.src_state.Xn[q_idx]

        cos = self.tgt_state.Xn @ qn
        cos = np.clip(cos, -1.0, 1.0)

        # ---- threshold filtering ----
        if similarity_threshold is not None:
            valid = np.where(cos >= similarity_threshold)[0]

            if len(valid) == 0:
                if not allow_fallback:
                    return [], None
                valid = np.argsort(cos)[::-1][:top_k]

            else:
                valid = valid[np.argsort(cos[valid])[::-1]]

        else:
            valid = np.argsort(cos)[::-1][:top_k]

        if max_top_k is not None:
            valid = valid[:max_top_k]

        indices = valid.tolist()
        if len(indices) == 0:
            return [], None

        best_bbox = self.topk_bbox(self.tgt_state, indices)

        top_patches = []
        for idx in indices:
            r, c = self.idx_to_rc(idx, self.tgt_state.cols)
            ps = self.tgt_state.ps
            top_patches.append(
                self.tgt_state.padded.crop(
                    (c * ps, r * ps, (c + 1) * ps, (r + 1) * ps)
                )
            )

        return top_patches, best_bbox


# ---------- Utility helpers (kept from your original file) ----------
def draw_bbox_on_image(image_path: str, output_path: str, bbox: Tuple[int, int, int, int], color='red', thickness=3):
    """
    Draw bounding box on image and save.
    """
    if bbox is None:
        raise ValueError("bbox is None; cannot draw.")

    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=thickness)

    # Save with format handling
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        image.save(output_path, 'JPEG', quality=95)
    else:
        image.save(output_path)
    print(f"Saved with bbox: {output_path}")
    print(f"BBox: ({x1}, {y1}) -> ({x2}, {y2})")


def resize_and_save_image(input_path: str, output_path: str, ratio: float = 1.5):
    """
    Load image, resize, and save with proper format handling.
    """
    image = Image.open(input_path)

    # Convert RGBA/LA/P to RGB for JPEG compatibility
    if image.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', image.size, (255, 255, 255))
        if image.mode == 'P':
            image = image.convert('RGBA')
        alpha = image.split()[-1] if image.mode in ('RGBA', 'LA') else None
        background.paste(image, mask=alpha)
        image = background

    width, height = image.size
    new_width = int(width / ratio)
    new_height = int(height / ratio)

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        resized_image = resized_image.convert('RGB')
        resized_image.save(output_path, 'JPEG', quality=95)
    else:
        resized_image.save(output_path)

    print(f"Saved: {output_path}")
    print(f"Original: {width}x{height} -> New: {new_width}x{new_height}")


# ---------- Main execution (glue) ----------
def main():
    # Model used by the DINO similarity component - keep same model id as before unless you want to change it
    dino_model_id = "./dinov3-convnext-tiny-pretrain-lvd1689m"
    sim = DINOv3PatchSimilarity(dino_model_id)

    # Input image that will be used as the source (where we detect the fire / prompt)
    input_path = "/home/ai/Desktop/PADDLE/fire_serv_test/frames/video_test_frame_003254.jpg"
    # Temporary preprocessed input (converted/resized as needed)
    input_rgb = "input.jpg"

    # Resize/convert the input image to avoid issues with formats
    resize_and_save_image(input_path, input_rgb, ratio=1)

    # Initialize user's SAM detector (uses default model_path in detector class)
    detector = SamDetector()

    # Load source image for detection (must be RGB for detector)
    src_image = Image.open(input_rgb).convert("RGB")

    # Use text prompt you want; here "small fire" as in your example
    prompt_text = "small fire"
    # Choose thresholds; adapt as you prefer
    threshold = 0.23
    mask_threshold = 0.5

    # Call your SAM-based detector
    print("Running detector on source image...")
    try:
        bboxes, centers = detector.detect(src_image, text=prompt_text, threshold=threshold, mask_threshold=mask_threshold, save_bboxes=False)
    except Exception as e:
        raise RuntimeError(f"Detector run failed: {e}")

    if not bboxes or not centers:
        print("No detections found by the SAM detector. Exiting.")
        return

    # Choose a center to use for DINO patch query.
    # If multiple centers returned, use the first selected center. You can change strategy here if needed.
    center_x, center_y = centers[0]
    # If detector returned integer bbox of selected region, keep it too (useful for visualization)
    detected_bbox = bboxes[0] if len(bboxes) > 0 else None

    print(f"Using detection center: ({center_x:.1f}, {center_y:.1f})")
    if detected_bbox:
        print(f"Detected source bbox: {detected_bbox}")

    # Iterate target frames in PATH and run the patch-similarity mapping
    frames_dir = "/home/ai/Desktop/PADDLE/fire_serv_test/frames"
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Optionally create output directory
    out_dir = "dino_results"
    os.makedirs(out_dir, exist_ok=True)

    for name in sorted(os.listdir(frames_dir)):
        img_path = os.path.join(frames_dir, name)

        # skip non-image files
        try:
            tgt_img = Image.open(img_path).convert("RGB")
        except Exception:
            print(f"Skipping non-image or unreadable file: {img_path}")
            continue

        # Process source and target images in DINO
        sim.process_images(src_image, tgt_img)

        # Query DINO with the center coordinates (x,y)
        # Note: SAM returns centers relative to the source image pixels; DINO expects the same
        top_patches, best_bbox = sim.select_patch(x=int(center_x), y=int(center_y))

        print(f"Frame: {name} -> Best bbox (on target image): {best_bbox}")

        # Draw the selected bbox (if any) on the target frame and save
        out_file = os.path.join(out_dir, f"annot_{name}")
        if best_bbox is not None:
            try:
                draw_bbox_on_image(img_path, out_file, best_bbox, color='red', thickness=3)
            except Exception as e:
                print(f"Failed to draw/save bbox for {img_path}: {e}")
        else:
            # If no best_bbox from DINO, optionally draw the detector bbox on the target (or copy)
            # Here we simply copy the target image to out_dir so you can see which frames had no mapping
            try:
                Image.open(img_path).convert("RGB").save(out_file)
                print(f"No DINO mapping for {name}; copied image to {out_file}")
            except Exception as e:
                print(f"Failed to save fallback copy for {img_path}: {e}")

    print("Finished processing frames.")


if __name__ == "__main__":
    main()
