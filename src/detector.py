import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
import os

os.environ["HF_TOKEN"] = "xxx"

class SamDetector:
    def __init__(self, model_path="./sam_model"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Sam3Model.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(self.device)
        self.processor = Sam3Processor.from_pretrained(model_path)

    def detect(self, image: Image.Image, text: str = "small fire", threshold: float = 0.5, mask_threshold: float = 0.5, save_bboxes: bool = False):
        """
        Perform promptable concept segmentation using SAM3 for detecting 'small fire' or other prompts.
        Computes bounding boxes from masks.
        - If multiple bboxes with confidence > 0.5, return all such bboxes and centers, and draw them on the image.
        - If none > 0.5, return the most confident bbox and center, and draw it.
        - If save_bboxes is True, save the image with drawn bboxes to 'detected_image.png'.
        Returns: tuple of (list of selected bboxes as tuples (x_min, y_min, x_max, y_max), list of centers as tuples (center_x, center_y))
        """
        if image is None:
            raise ValueError("Please provide an image.")
        
        if not text.strip():
            raise ValueError("Please enter a text prompt.")
        
        try:
            inputs = self.processor(images=image, text=text.strip(), return_tensors="pt").to(self.device)
            
            for key in inputs:
                if inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].to(self.model.dtype)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            n_masks = len(results['masks'])
            if n_masks == 0:
                print(f"No objects found matching '{text}' (try adjusting thresholds).")
                return [], []
            
            # Compute bounding boxes and scores
            bboxes = []
            scores = results['scores'].cpu().numpy()
            for mask in results['masks']:
                mask_np = mask.cpu().numpy()
                rows = np.any(mask_np > 0, axis=1)
                cols = np.any(mask_np > 0, axis=0)
                if rows.sum() == 0:
                    continue
                y_min = np.where(rows)[0].min()
                y_max = np.where(rows)[0].max() + 1
                x_min = np.where(cols)[0].min()
                x_max = np.where(cols)[0].max() + 1
                bboxes.append((x_min, y_min, x_max, y_max))
            
            # Select bboxes
            selected_bboxes = []
            selected_indices = []
            high_conf_indices = [i for i, s in enumerate(scores) if s > 0.5]
            
            if len(high_conf_indices) > 0:
                # Multiple high confidence
                selected_indices = high_conf_indices
            else:
                # Most confident
                if len(scores) > 0:
                    max_idx = np.argmax(scores)
                    selected_indices = [max_idx]
            
            selected_bboxes = [bboxes[i] for i in selected_indices]
            selected_scores = [scores[i] for i in selected_indices]
            
            if not selected_bboxes:
                print("No bounding boxes selected.")
                return [], []
            
            # Compute centers
            centers = [((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) for bbox in selected_bboxes]
            
            # Draw on image copy
            image_out = image.copy()
            draw = ImageDraw.Draw(image_out)
            for i, (bbox, score) in enumerate(zip(selected_bboxes, selected_scores)):
                draw.rectangle(bbox, outline="red", width=3)
                text_label = f"#{i+1} ({score:.2f})"
                text_y = bbox[1] - 15 if bbox[1] > 15 else bbox[3] + 5
                draw.text((bbox[0], text_y), text_label, fill="red")
            
            # Save if requested
            if save_bboxes:
                output_path = "detected_image.png"
                image_out.save(output_path)
                print(f"Image with bboxes saved to {output_path}")
            
            # Print info
            print(f"Found {len(selected_bboxes)} selected bounding boxes for '{text}'")
            for i, (bbox, center, score) in enumerate(zip(selected_bboxes, centers, selected_scores)):
                print(f"#{i+1}: (left={bbox[0]}, top={bbox[1]}, right={bbox[2]}, bottom={bbox[3]}) (center=({center[0]:.2f}, {center[1]:.2f})) (score: {score:.2f})")
            
            return selected_bboxes, centers
        
        except Exception as e:
            raise RuntimeError(f"Error during detection: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Load an example image (replace with your image path or URL)
    import requests
    image_path = "fire.png"  # Example, replace with fire image if needed
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    
    # Initialize detector
    detector = SamDetector()
    
    # Run detection
    bboxes, centers = detector.detect(image, text="small fire", threshold=0.23, mask_threshold=0.5, save_bboxes=True)
    print("Returned bboxes:", bboxes)
    print("Returned centers:", centers)
