# main.py
import os
import io
from typing import Optional, List, Tuple
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw
import numpy as np
import torch

import os

os.environ["HF_TOKEN"] = "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"

if not os.environ.get("HF_TOKEN"):
    raise RuntimeError("Please set HF_TOKEN environment variable (HuggingFace token).")

# Import transformers after token is present
from transformers import Sam3Processor, Sam3Model

# ---- SamDetector class (adapted for robust use in API) ----
class SamDetector:
    def __init__(self, model_path: str = "./sam_model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if (torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 0) else torch.float32
        # Load model and processor
        self.model = Sam3Model.from_pretrained(model_path, torch_dtype=dtype).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_path)

    def detect(self, image: Image.Image, text: str = "small fire", threshold: float = 0.23, mask_threshold: float = 0.5, save_bboxes: bool = False):
        """
        Returns (selected_bboxes, centers, selected_scores, annotated_image_pil)
        selected_bboxes: list of tuples (x_min, y_min, x_max, y_max)
        centers: list of tuples (center_x, center_y)
        selected_scores: list of float
        annotated_image_pil: PIL.Image with drawn bboxes (or None if something failed)
        """
        if image is None:
            raise ValueError("Please provide an image.")
        if not text or not text.strip():
            raise ValueError("Please provide a non-empty text prompt.")

        try:
            # Prepare inputs (keep original sizes separately)
            inputs = self.processor(images=image, text=text.strip(), return_tensors="pt")
            original_sizes = inputs.pop("original_sizes").tolist() if "original_sizes" in inputs else None

            # Move tensors to device and to model dtype
            model_dtype = self.model.dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            for k in list(inputs.keys()):
                if torch.is_tensor(inputs[k]) and inputs[k].dtype == torch.float32:
                    inputs[k] = inputs[k].to(model_dtype)

            with torch.no_grad():
                outputs = self.model(**inputs)

            post_kwargs = {"threshold": threshold, "mask_threshold": mask_threshold}
            if original_sizes is not None:
                post_kwargs["target_sizes"] = original_sizes

            results = self.processor.post_process_instance_segmentation(outputs, **post_kwargs)[0]

            masks = results.get("masks", [])
            scores = results.get("scores", torch.tensor([])).cpu().numpy() if "scores" in results else np.array([])

            if len(masks) == 0:
                return [], [], [], None

            # Compute bounding boxes from masks
            bboxes = []
            for mask in masks:
                mask_np = mask.cpu().numpy()
                rows = np.any(mask_np > 0, axis=1)
                cols = np.any(mask_np > 0, axis=0)
                if rows.sum() == 0 or cols.sum() == 0:
                    # empty mask
                    bboxes.append((0,0,0,0))
                    continue
                y_min = int(np.where(rows)[0].min())
                y_max = int(np.where(rows)[0].max()) + 1
                x_min = int(np.where(cols)[0].min())
                x_max = int(np.where(cols)[0].max()) + 1
                bboxes.append((x_min, y_min, x_max, y_max))

            # Select bboxes based on scores
            selected_indices = [i for i, s in enumerate(scores) if s > 0.5]
            if not selected_indices:
                if len(scores) > 0:
                    selected_indices = [int(np.argmax(scores))]
                else:
                    selected_indices = list(range(len(bboxes)))  # fallback

            selected_bboxes = [bboxes[i] for i in selected_indices]
            selected_scores = [float(scores[i]) if i < len(scores) else 0.0 for i in selected_indices]
            centers = [((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0) for bbox in selected_bboxes]

            # Annotate image
            image_out = image.copy()
            draw = ImageDraw.Draw(image_out)
            for i, (bbox, score) in enumerate(zip(selected_bboxes, selected_scores)):
                x_min, y_min, x_max, y_max = bbox
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                label = f"#{i+1} {score:.2f}"
                text_y = y_min - 16 if y_min > 16 else y_max + 2
                draw.text((x_min, text_y), label, fill="red")

            if save_bboxes:
                image_out.save("detected_image.png")

            return selected_bboxes, centers, selected_scores, image_out

        except Exception as e:
            raise RuntimeError(f"Error during detection: {e}")


# ---- FastAPI app ----
app = FastAPI(title="SAM Detector API")

# Create a single global detector instance (loaded on first import / startup)
DETECTOR: Optional[SamDetector] = None

@app.on_event("startup")
def startup_event():
    global DETECTOR
    if DETECTOR is None:
        model_path = os.getenv("SAM_MODEL_PATH", "./sam_model")
        DETECTOR = SamDetector(model_path=model_path)

# ---- Request models ----
class DetectResponse(BaseModel):
    bboxes: List[Tuple[int, int, int, int]]
    centers: List[Tuple[float, float]]
    scores: List[float]

# ---- Helper to load image from UploadFile or URL ----
def load_image_from_upload_or_url(upload_file: Optional[UploadFile], image_url: Optional[str]) -> Image.Image:
    if upload_file is None and (image_url is None or image_url.strip() == ""):
        raise HTTPException(status_code=400, detail="Provide either an image file or image_url.")
    if upload_file:
        contents = upload_file.file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot read uploaded image: {e}")
    else:
        # load from URL (optional; requires 'requests')
        try:
            import requests
            resp = requests.get(image_url, stream=True, timeout=15)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return image
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot fetch image from URL: {e}")

# ---- Endpoints ----
@app.post("/detect", response_model=DetectResponse)
def detect_endpoint(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text: str = Form("small fire"),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5),
):
    """
    Return JSON with bboxes, centers, and scores.
    - image: file upload (multipart/form-data)
    - image_url: optional URL to fetch image
    - text: prompt (default: "small fire")
    - threshold: detection threshold (default 0.5)
    - mask_threshold: mask binarization threshold (default 0.5)
    """
    global DETECTOR
    if DETECTOR is None:
        raise HTTPException(status_code=500, detail="Detector not initialized.")
    pil_image = load_image_from_upload_or_url(image, image_url)
    try:
        bboxes, centers, scores, _ = DETECTOR.detect(pil_image, text=text, threshold=threshold, mask_threshold=mask_threshold, save_bboxes=False)
        return DetectResponse(bboxes=bboxes, centers=centers, scores=scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-image")
def detect_image_endpoint(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text: str = Form("small fire"),
    threshold: float = Form(0.5),
    mask_threshold: float = Form(0.5),
    save_bboxes: bool = Form(False),
):
    """
    Return the image with drawn bounding boxes (PNG).
    Accepts same inputs as /detect.
    """
    global DETECTOR
    if DETECTOR is None:
        raise HTTPException(status_code=500, detail="Detector not initialized.")
    pil_image = load_image_from_upload_or_url(image, image_url)
    try:
        bboxes, centers, scores, annotated = DETECTOR.detect(pil_image, text=text, threshold=threshold, mask_threshold=mask_threshold, save_bboxes=save_bboxes)
        if annotated is None:
            # return original image if nothing detected
            annotated = pil_image
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ---- optionally allow running with `python main.py` ----
if __name__ == "__main__":
    import uvicorn
    # Example: uvicorn main:app --host 0.0.0.0 --port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
