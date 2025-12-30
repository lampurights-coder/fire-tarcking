# import spaces
# import gradio as gr
# import torch
# import numpy as np
# from PIL import Image, ImageDraw
# from transformers import Sam3Processor, Sam3Model
# import requests
# import warnings
# warnings.filterwarnings("ignore")
# import os
# os.environ["HF_TOKEN"] = "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"
# # Global model and processor
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = Sam3Model.from_pretrained("/home/ai/Desktop/PADDLE/fire_serv_test/sam_3/sam_model", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
# processor = Sam3Processor.from_pretrained("/home/ai/Desktop/PADDLE/fire_serv_test/sam_3/sam_model")

# @spaces.GPU()
# def segment(image: Image.Image, text: str, threshold: float, mask_threshold: float, draw_bboxes: bool = False):
#     """
#     Perform promptable concept segmentation using SAM3.
#     Returns format compatible with gr.AnnotatedImage: (image, [(mask, label), ...])
#     Additionally computes bounding boxes from masks and optionally draws them on the image.
#     """
#     if image is None:
#         return None, "❌ Please upload an image.", ""
    
#     if not text.strip():
#         return (image, []), "❌ Please enter a text prompt.", ""
    
#     try:
#         inputs = processor(images=image, text=text.strip(), return_tensors="pt").to(device)
        
#         for key in inputs:
#             if inputs[key].dtype == torch.float32:
#                 inputs[key] = inputs[key].to(model.dtype)
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         results = processor.post_process_instance_segmentation(
#             outputs,
#             threshold=threshold,
#             mask_threshold=mask_threshold,
#             target_sizes=inputs.get("original_sizes").tolist()
#         )[0]
        
#         n_masks = len(results['masks'])
#         if n_masks == 0:
#             return (image, []), f"❌ No objects found matching '{text}' (try adjusting thresholds).", "No bounding boxes found."
        
#         # Compute bounding boxes from masks
#         bboxes = []
#         for mask in results['masks']:
#             mask_np = mask.cpu().numpy()
#             rows = np.any(mask_np > 0, axis=1)
#             cols = np.any(mask_np > 0, axis=0)
#             if rows.sum() == 0:
#                 continue
#             y_min = np.where(rows)[0].min()
#             y_max = np.where(rows)[0].max() + 1
#             x_min = np.where(cols)[0].min()
#             x_max = np.where(cols)[0].max() + 1
#             bboxes.append((x_min, y_min, x_max, y_max))
        
#         # Format bboxes string
#         bboxes_str = "\n".join([
#             f"#{i+1}: (left={bbox[0]}, top={bbox[1]}, right={bbox[2]}, bottom={bbox[3]}) (score: {score:.2f})"
#             for i, (bbox, score) in enumerate(zip(bboxes, results['scores'].cpu().numpy()))
#         ])
        
#         scores_text = ", ".join([f"{s:.2f}" for s in results['scores'].cpu().numpy()[:5]])
#         info = f"✅ Found **{n_masks}** objects matching **'{text}'**\nConfidence scores: {scores_text}{'...' if n_masks > 5 else ''}"
        
#         if draw_bboxes:
#             # Draw bboxes on a copy of the image
#             image_out = image.copy()
#             draw = ImageDraw.Draw(image_out)
#             for i, (bbox, score) in enumerate(zip(bboxes, results['scores'])):
#                 draw.rectangle(bbox, outline="red", width=3)
#                 text_label = f"#{i+1} ({score:.2f})"
#                 text_y = bbox[1] - 15 if bbox[1] > 15 else bbox[3] + 5
#                 draw.text((bbox[0], text_y), text_label, fill="red")
#             # Return drawn image with no mask annotations
#             return (image_out, []), info, bboxes_str
#         else:
#             # Format for AnnotatedImage: list of (mask, label) tuples
#             annotations = [
#                 (mask.cpu().numpy().astype(np.float32), f"{text} #{i+1} ({score:.2f})")
#                 for i, (mask, score) in enumerate(zip(results['masks'], results['scores']))
#             ]
#             return (image, annotations), info, bboxes_str
        
#     except Exception as e:
#         return (image, []), f"❌ Error during segmentation: {str(e)}", ""

# def clear_all():
#     """Clear all inputs and outputs"""
#     return None, "", None, 0.5, 0.5, "📝 Enter a prompt and click **Segment** to start.", False, ""

# def segment_example(image_path: str, prompt: str):
#     """Handle example clicks"""
#     if image_path.startswith("http"):
#         image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
#     else:
#         image = Image.open(image_path).convert("RGB")
#     return segment(image, prompt, 0.5, 0.5, False)

# # Gradio Interface
# with gr.Blocks(
#     theme=gr.themes.Soft(),
#     title="SAM3 - Promptable Concept Segmentation",
#     css=".gradio-container {max-width: 1400px !important;}"
# ) as demo:
#     gr.Markdown(
#         """
#         # SAM3 - Promptable Concept Segmentation (PCS)
        
#         **SAM3** performs zero-shot instance segmentation using natural language prompts.
#         Upload an image, enter a text prompt (e.g., "person", "car", "dog"), and get segmentation masks.
        
#         Built with [anycoder](https://huggingface.co/spaces/akhaliq/anycoder)
#         """
#     )
    
#     gr.Markdown("### Inputs")
#     with gr.Row(variant="panel"):
#         image_input = gr.Image(
#             label="Input Image",
#             type="pil",
#             height=400,
#         )
#         # AnnotatedImage expects: (base_image, [(mask, label), ...])
#         image_output = gr.AnnotatedImage(
#             label="Output (Segmented Image)",
#             height=400,
#             show_legend=True,
#         )
    
#     with gr.Row():
#         text_input = gr.Textbox(
#             label="Text Prompt",
#             placeholder="e.g., person, ear, cat, bicycle...",
#             scale=3
#         )
#         clear_btn = gr.Button("🔍 Clear", size="sm", variant="secondary")
    
#     with gr.Row():
#         thresh_slider = gr.Slider(
#             minimum=0.0,
#             maximum=1.0,
#             value=0.5,
#             step=0.01,
#             label="Detection Threshold",
#             info="Higher = fewer detections"
#         )
#         mask_thresh_slider = gr.Slider(
#             minimum=0.0,
#             maximum=1.0,
#             value=0.5,
#             step=0.01,
#             label="Mask Threshold",
#             info="Higher = sharper masks"
#         )
    
#     draw_bboxes = gr.Checkbox(
#         label="Draw Bounding Boxes (instead of masks)",
#         value=False,
#         info="If checked, draws bounding boxes on the image instead of overlaying masks."
#     )
    
#     info_output = gr.Markdown(
#         value="📝 Enter a prompt and click **Segment** to start.",
#         label="Info / Results"
#     )
    
#     bbox_output = gr.Textbox(
#         label="Bounding Boxes",
#         lines=5,
#         interactive=False
#     )
    
#     segment_btn = gr.Button("🎯 Segment", variant="primary", size="lg")
    
#     gr.Examples(
#         examples=[
#             ["http://images.cocodataset.org/val2017/000000077595.jpg", "cat"],
#         ],
#         inputs=[image_input, text_input],
#         outputs=[image_output, info_output, bbox_output],
#         fn=segment_example,
#         cache_examples=False,
#     )
    
#     clear_btn.click(
#         fn=clear_all,
#         outputs=[image_input, text_input, image_output, thresh_slider, mask_thresh_slider, info_output, draw_bboxes, bbox_output]
#     )
    
#     segment_btn.click(
#         fn=segment,
#         inputs=[image_input, text_input, thresh_slider, mask_thresh_slider, draw_bboxes],
#         outputs=[image_output, info_output, bbox_output]
#     )
    
#     gr.Markdown(
#         """
#         ### Notes
#         - **Model**: [facebook/sam3](https://huggingface.co/facebook/sam3)
#         - Click on segments in the output to see labels
#         - GPU recommended for faster inference
#         """
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)

import spaces
import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model
import requests
import warnings
warnings.filterwarnings("ignore")
import os

os.environ["HF_TOKEN"] = "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"

# Global model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained(
    "/home/ai/Desktop/PADDLE/fire_serv_test/sam_3/sam_model",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

processor = Sam3Processor.from_pretrained(
    "/home/ai/Desktop/PADDLE/fire_serv_test/sam_3/sam_model"
)

@spaces.GPU()
def segment(image: Image.Image, text: str, threshold: float, mask_threshold: float, draw_bboxes: bool = False):
    """
    Perform promptable concept segmentation using SAM3.
    Returns format compatible with gr.AnnotatedImage: (image, [(mask, label), ...])
    Additionally computes bounding boxes from masks and returns max-confidence bbox.
    """

    if image is None:
        return None, "❌ Please upload an image.", ""
    
    if not text.strip():
        return (image, []), "❌ Please enter a text prompt.", ""
    
    try:
        inputs = processor(images=image, text=text.strip(), return_tensors="pt").to(device)

        # cast float32 → model dtype if needed
        for key in inputs:
            if inputs[key].dtype == torch.float32:
                inputs[key] = inputs[key].to(model.dtype)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        n_masks = len(results['masks'])

        if n_masks == 0:
            return (image, []), f"❌ No objects found matching '{text}' (try adjusting thresholds).", "No bounding boxes found."
        
        # -------- COMPUTE BBOXES ----------
        bboxes = []
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

        scores = results['scores'].cpu().numpy()

        # -------- FIND MAX CONFIDENCE ----------
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        max_bbox = bboxes[max_idx]

        max_conf_str = (
            f"MAX: (left={max_bbox[0]}, top={max_bbox[1]}, "
            f"right={max_bbox[2]}, bottom={max_bbox[3]}) "
            f"(score: {max_score:.4f})"
        )

        # -------- FORMAT ALL BBOXES ----------
        bboxes_str = "\n".join([
            f"#{i+1}: (left={bbox[0]}, top={bbox[1]}, right={bbox[2]}, bottom={bbox[3]}) (score: {score:.4f})"
            for i, (bbox, score) in enumerate(zip(bboxes, scores))
        ])

        # append max bbox summary
        bboxes_str = bboxes_str + "\n\n" + max_conf_str

        scores_text = ", ".join([f"{s:.2f}" for s in scores[:5]])
        info = (
            f"✅ Found **{n_masks}** objects matching **'{text}'**\n"
            f"Confidence scores: {scores_text}{'...' if n_masks > 5 else ''}\n\n"
            f"🔥 **Highest confidence detection:** {max_score:.4f}"
        )

        # -------- DRAW BBOXES MODE ----------
        if draw_bboxes:
            image_out = image.copy()
            draw = ImageDraw.Draw(image_out)

            for i, (bbox, score) in enumerate(zip(bboxes, scores)):
                draw.rectangle(bbox, outline="red", width=3)
                label = f"#{i+1} ({score:.2f})"
                text_y = bbox[1] - 15 if bbox[1] > 15 else bbox[3] + 5
                draw.text((bbox[0], text_y), label, fill="red")

            # highlight max bbox in green
            draw.rectangle(max_bbox, outline="green", width=4)
            draw.text((max_bbox[0], max_bbox[1] - 18), "MAX", fill="green")

            return (image_out, []), info, bboxes_str

        # -------- MASK OVERLAY MODE ----------
        annotations = [
            (mask.cpu().numpy().astype(np.float32), f"{text} #{i+1} ({score:.2f})")
            for i, (mask, score) in enumerate(zip(results['masks'], scores))
        ]

        return (image, annotations), info, bboxes_str

    except Exception as e:
        return (image, []), f"❌ Error during segmentation: {str(e)}", ""

def clear_all():
    return None, "", None, 0.5, 0.5, "📝 Enter a prompt and click **Segment** to start.", False, ""

def segment_example(image_path: str, prompt: str):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return segment(image, prompt, 0.5, 0.5, False)


# ------------------ GRADIO UI ------------------

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="SAM3 - Promptable Concept Segmentation",
    css=".gradio-container {max-width: 1400px !important;}"
) as demo:

    gr.Markdown(
        """
        # SAM3 - Promptable Concept Segmentation (PCS)
        Upload an image + text prompt to get segmentation masks / bounding boxes.
        """
    )

    gr.Markdown("### Inputs")

    with gr.Row(variant="panel"):
        image_input = gr.Image(label="Input Image", type="pil", height=400)

        image_output = gr.AnnotatedImage(
            label="Output (Segmented Image)",
            height=400,
            show_legend=True,
        )

    with gr.Row():
        text_input = gr.Textbox(
            label="Text Prompt",
            placeholder="e.g., person, car, dog",
            scale=3
        )
        clear_btn = gr.Button("🔍 Clear", size="sm", variant="secondary")

    with gr.Row():
            thresh_slider = gr.Slider(0.0, 1.0, 0.5, 0.01, label="Detection Threshold")
            mask_thresh_slider = gr.Slider(0.0, 1.0, 0.5, 0.01, label="Mask Threshold")

    draw_bboxes = gr.Checkbox(
        label="Draw Bounding Boxes (instead of masks)",
        value=False
    )

    info_output = gr.Markdown(
        value="📝 Enter a prompt and click **Segment** to start.",
        label="Info / Results"
    )

    bbox_output = gr.Textbox(
        label="Bounding Boxes (includes MAX bbox)",
        lines=6,
        interactive=False
    )

    segment_btn = gr.Button("🎯 Segment", variant="primary", size="lg")

    gr.Examples(
        examples=[
            ["http://images.cocodataset.org/val2017/000000077595.jpg", "cat"],
        ],
        inputs=[image_input, text_input],
        outputs=[image_output, info_output, bbox_output],
        fn=segment_example,
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[image_input, text_input, image_output,
                 thresh_slider, mask_thresh_slider,
                 info_output, draw_bboxes, bbox_output]
    )

    segment_btn.click(
        fn=segment,
        inputs=[image_input, text_input, thresh_slider,
                mask_thresh_slider, draw_bboxes],
        outputs=[image_output, info_output, bbox_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)
