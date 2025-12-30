#!/usr/bin/env python3
"""
gradio_video.py

Real-time Fire Detection & Tracking UI using:
 - SamDetector (promptable SAM3 based detector) from src.detector
 - DINOv3 patch similarity from src.dino
 - BotSort for multi-object tracking
 - FrameSynchronizer from src.frame_gather for synchronized frame retrieval

Drop this file into your project and run it. Make sure:
 - src/detector.py contains SamDetector class with .detect(image, text, threshold, mask_threshold, save_bboxes)
 - src/dino.py contains DINOv3PatchSimilarity
 - src/frame_gather.py contains FrameSynchronizer
"""

import cv2
import time
import threading
import os
import yaml
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import gradio as gr

# Project imports (must exist in your repo)
from src.frame_gather import FrameSynchronizer
from src.dino import DINOv3PatchSimilarity
from src.detector import SamDetector
from boxmot import BotSort


class FireDINOTrackerGradio:
    def __init__(
        self,
        config_path="./configs/config.yaml",
        fps=20,
    ):
        self.config_path = config_path
        self.fps = fps

        # Detection executor / futures
        self.detect_future = None
        self.detect_frame = None
        self.executor = None

        # DINO similarity model
        self.sim = DINOv3PatchSimilarity("./dinov3-convnext-tiny-pretrain-lvd1689m")

        # Reference frame (captured from detection) and center
        self.ref_frame_path = None
        self.ref_center = None
        self.lock = threading.Lock()

        # Instantiate SAM-based detector once
        self.detector = SamDetector()

        # BotSort tracker (will be re-initialized per session)
        self.tracker = BotSort(
            reid_weights="",
            device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            fp16=False,
            with_reid=False,
            half=True
        )

        # Timing / control
        self.last_submit_time = 0.0
        self.vlm_interval = 1  # detection interval in seconds (adjustable)
        self.stop_processing = False
        self.video_writer = None

    # ---------------- Config helpers ----------------
    def load_config(self):
        if not os.path.exists(self.config_path):
            return {}
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f) or {}

    def update_config_video_path(self, video_path):
        config = self.load_config()
        if config.get('sources') and len(config['sources']) > 0:
            config['sources'][0]['url'] = video_path
        else:
            config['sources'] = [{'id': 11, 'url': video_path}]
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config

    def get_video_path_from_config(self):
        config = self.load_config()
        if config.get('sources') and len(config['sources']) > 0:
            return config['sources'][0].get('url')
        return None

    # ---------------- Detection (SAM) submission & handling ----------------
    def _submit_detection(self, frame):
        """
        Submit a detection job to the executor using the SAM detector.
        Frame must be BGR numpy (as from OpenCV).
        """
        # Save detect_frame for use with BotSort update call which expects a frame
        self.detect_frame = frame.copy()
        # Convert to PIL RGB for detector
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Submit the SamDetector.detect call with typical prompt/thresholds
        return self.executor.submit(self.detector.detect, pil_img, "small fire", 0.2, 0.5, False)

    def _handle_detection_result(self):
        """
        Check detect_future; if completed, process results and set reference frame & center.
        """
        if not (self.detect_future and self.detect_future.done()):
            return

        try:
            result = self.detect_future.result()
            # SamDetector.detect -> returns (selected_bboxes, centers)
            if not result or not isinstance(result, tuple) or len(result) != 2:
                print("[DETECT] Invalid detector result")
                return

            bboxes, centers = result

            if not bboxes or not centers:
                print("[DETECT] no fire")
                return

            # Convert SAM bboxes to BotSort detection format: [x1, y1, x2, y2, conf, cls]
            dets = np.array([[*b, 1.0, 0] for b in bboxes], dtype=np.float32)
            # Update BotSort with detections detected on the detect_frame
            self.tracker.update(dets, self.detect_frame)

            # Choose the largest bbox (fallback: first)
            areas = [ (b[2]-b[0])*(b[3]-b[1]) for b in bboxes ]
            idx = int(np.argmax(areas)) if len(areas) > 0 else 0
            x, y = centers[idx]

            print(f"[DETECT] fire center: ({x:.1f}, {y:.1f})")

            ts = int(time.time() * 1000)
            ref_path = f"/tmp/fire_ref_{ts}.jpg"
            cv2.imwrite(ref_path, self.detect_frame)

            with self.lock:
                self.ref_frame_path = ref_path
                # Save integer coords for DINO
                self.ref_center = (int(round(x)), int(round(y)))

        except Exception as e:
            print("[DETECT ERROR]", e)

        finally:
            # reset futures
            self.detect_future = None
            self.detect_frame = None

    # ---------------- Video writer helper ----------------
    def _init_video_writer(self, frame, output_path):
        if self.video_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
            print(f"[INFO] Recording to {output_path}")

    # ---------------- Per-frame processing ----------------
    def process_frame(self, frame):
        """
        Process single BGR frame: submit periodic detection, run DINO mapping if ref set,
        update BotSort, annotate and return frame (BGR).
        """
        current_time = time.time()

        # Submit detection periodically
        if self.detect_future is None and (current_time - self.last_submit_time) > self.vlm_interval:
            self.detect_future = self._submit_detection(frame)
            self.last_submit_time = current_time

        # Handle completed detection
        self._handle_detection_result()

        # Use reference (if any) to run DINO patch similarity for mapping
        with self.lock:
            ref_path = self.ref_frame_path
            ref_center = self.ref_center

        if ref_path and ref_center:
            # Save current frame temporarily (DINO expects PIL images)
            cur_path = "/tmp/current_frame.jpg"
            cv2.imwrite(cur_path, frame)

            try:
                src_img = Image.open(ref_path).convert("RGB")
                tgt_img = Image.open(cur_path).convert("RGB")

                self.sim.process_images(src_img, tgt_img)
                x, y = ref_center
                patches, bbox = self.sim.select_patch(x=x, y=y)

                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        "FIRE (DINO)",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )
            except Exception as e:
                print("[DINO ERROR]", e)

        # BotSort tracking update with no fresh detections (we rely on stored tracker state)
        empty_dets = np.empty((0, 6), dtype=np.float32)
        try:
            tracks = self.tracker.update(empty_dets, frame)
        except Exception as e:
            print("[TRACKER ERROR]", e)
            tracks = []

        # Annotate tracks
        for trk in tracks:
            # BotSort track format might differ; adapt if necessary
            try:
                x1, y1, x2, y2, trk_id, conf, cls_, *_ = trk
            except Exception:
                # safe fallback: try first 6
                vals = list(trk)
                if len(vals) >= 6:
                    x1, y1, x2, y2, trk_id, conf = vals[:6]
                    cls_ = 0
                else:
                    continue
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"FIRE (BOT-SORT ID:{int(trk_id)})",
                (x1, max(20, y1 - 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )

        # Status overlay if nothing yet
        if (not ref_path) and len(tracks) == 0:
            cv2.putText(
                frame,
                "DETECTING FIRE...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        return frame

    # ---------------- Video stream processing generator ----------------
    def process_video_stream(self, video_path=None):
        """
        Generator that yields processed frames and status messages for Gradio.
        Uses FrameSynchronizer to pull frames synchronized from configured sources.
        """
        # Create an executor for detection jobs
        self.executor = ThreadPoolExecutor(max_workers=1)

        # If a video_path is supplied (from the UI), update config
        if video_path:
            self.update_config_video_path(video_path)
            current_video = video_path
        else:
            current_video = self.get_video_path_from_config()

        if not current_video:
            yield None, "❌ Video source not specified!"
            return

        is_rtsp = str(current_video).startswith("rtsp://")
        if (not is_rtsp) and (not os.path.exists(current_video)):
            yield None, "❌ Video file not found!"
            return

        # Reset internal state
        self.stop_processing = False
        self.ref_frame_path = None
        self.ref_center = None
        # Re-init tracker for this session
        self.tracker = BotSort(
            reid_weights="",
            device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
            fp16=False,
            with_reid=False,
            half=True
        )

        # Output file
        output_path = f"./output/fire_tracking_{int(time.time())}.mp4"
        os.makedirs(Path(output_path).parent, exist_ok=True)

        # Probe video to get fps/frames (if local file)
        temp_cap = cv2.VideoCapture(current_video) if not is_rtsp else cv2.VideoCapture(current_video)
        if not temp_cap.isOpened():
            yield None, "❌ Failed to open video source!"
            return
        try:
            total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_rtsp else 0
            fps = temp_cap.get(cv2.CAP_PROP_FPS) or self.fps
            self.fps = fps
        finally:
            temp_cap.release()

        # Start FrameSynchronizer and processing loop
        sync = FrameSynchronizer(self.config_path)
        sync.start()

        frame_count = 0
        try:
            while True:
                if self.stop_processing:
                    yield None, "⏹️ Processing stopped by user"
                    break

                data = sync.get_synchronized()
                if not data:
                    time.sleep(0.01)
                    continue

                frames, _ = data
                if not frames:
                    # no frames left
                    break

                frame, cam = frames[0]

                # Initialize video writer once
                self._init_video_writer(frame, output_path)

                # Process this frame
                processed_frame = self.process_frame(frame.copy())

                # Write processed frame to output file
                if self.video_writer:
                    self.video_writer.write(processed_frame)

                # Convert BGR -> RGB for Gradio display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                frame_count += 1
                if total_frames > 0:
                    status = f"🔥 Processing: {frame_count}/{total_frames} frames ({(frame_count/total_frames*100):.1f}%)"
                else:
                    status = f"🔥 Processing live stream: {frame_count} frames"

                # Yield to Gradio
                yield processed_frame_rgb, status

                
                

        finally:
            # Clean up
            sync.stop()
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None

            yield None, f"✅ Complete! Processed {frame_count} frames. Saved to: {output_path}"

    # ---------------- Control / UI helpers ----------------
    def stop(self):
        """Stop background processing"""
        self.stop_processing = True
        return "⏹️ Stopping..."

    def upload_video_to_config(self, video_file):
        """Save uploaded video and update config path"""
        if video_file is None:
            return "❌ No video selected", None
        video_dir = "./uploaded_videos"
        os.makedirs(video_dir, exist_ok=True)
        file_name = Path(video_file.name).name
        dest_path = os.path.join(video_dir, f"{int(time.time())}_{file_name}")
        # video_file is a tempfile with .name property; copy it
        shutil.copy(video_file.name, dest_path)
        self.update_config_video_path(dest_path)
        return f"✅ Video uploaded and config updated!\nPath: {dest_path}", dest_path

    def set_rtsp_to_config(self, rtsp_url):
        if not rtsp_url or not rtsp_url.startswith('rtsp://'):
            return "❌ Invalid RTSP URL! Must start with 'rtsp://'", None
        self.update_config_video_path(rtsp_url)
        return f"✅ RTSP URL set and config updated!\nURL: {rtsp_url}", rtsp_url


# ---------------- Gradio UI ----------------
def create_gradio_interface():
    tracker = FireDINOTrackerGradio(config_path="./configs/config.yaml")

    with gr.Blocks(title="Fire DINO Tracker - Real-time", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔥 Fire Detection & Tracking System (Real-time)

        Upload a video file or enter an RTSP URL. Processing happens in **real-time** with live preview.

        ### Features:
        - **SAM Detection**: promptable SAM detects fire (every few seconds)
        - **DINO Tracking**: DINOv3 patch similarity tracks fire across frames
        - **BotSort**: Multi-object tracking with consistent IDs
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Video Source")

                with gr.Tab("Upload File"):
                    video_upload = gr.File(
                        label="Upload Video File",
                        file_types=["video"],
                        type="filepath"
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        lines=2
                    )
                    upload_btn = gr.Button("📤 Upload & Update Config", variant="primary", size="lg")

                with gr.Tab("RTSP URL"):
                    rtsp_input = gr.Textbox(
                        label="Enter RTSP URL (e.g., rtsp://example.com/stream)",
                        placeholder="rtsp://..."
                    )
                    rtsp_status = gr.Textbox(
                        label="RTSP Status",
                        interactive=False,
                        lines=2
                    )
                    rtsp_btn = gr.Button("🔗 Set RTSP URL", variant="primary", size="lg")

                gr.Markdown("---")

                with gr.Row():
                    start_btn = gr.Button("▶️ Start Processing", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop")

                status_text = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=2
                )

                gr.Markdown("""
                ### ⚙️ Current Config
                Config file: `./configs/config.yaml`

                Video path/URL is read from config and updated when you upload a file or set RTSP.
                """)

            with gr.Column(scale=2):
                gr.Markdown("### 🎥 Live Processing View")

                live_output = gr.Image(
                    label="Real-time Fire Detection",
                    type="numpy",
                    height=600,
                    show_label=True
                )

                gr.Markdown("""
                ### 💡 How it works:
                1. **Upload** a video file or **enter RTSP URL** (updates config automatically)
                2. Click **Start Processing** to begin real-time detection
                3. Watch live as each frame is processed
                4. Detection runs every few seconds (configurable in code)
                5. Processed video is saved to `./output/` directory
                6. Use **Stop** button to cancel processing anytime
                """)

        # Hidden state for video path
        video_path_state = gr.State(value=None)

        # Upload handler
        def upload_and_update(video_file):
            status, path = tracker.upload_video_to_config(video_file)
            return status, path

        upload_btn.click(
            fn=upload_and_update,
            inputs=[video_upload],
            outputs=[upload_status, video_path_state]
        )

        # RTSP handler
        def set_rtsp_and_update(rtsp_url):
            status, path = tracker.set_rtsp_to_config(rtsp_url)
            return status, path

        rtsp_btn.click(
            fn=set_rtsp_and_update,
            inputs=[rtsp_input],
            outputs=[rtsp_status, video_path_state]
        )

        # Start processing (generator)
        start_btn.click(
            fn=tracker.process_video_stream,
            inputs=[video_path_state],
            outputs=[live_output, status_text]
        )

        stop_btn.click(
            fn=tracker.stop,
            outputs=[status_text]
        )

        gr.Markdown("""
        ---
        ### 📝 Notes:
        - Processing shows frames in **real-time** as they're processed (not after completion)
        - Original video path in config: will be used if no video is uploaded or RTSP set
        - Output videos are saved to: `./output/fire_tracking_[timestamp].mp4`
        - Detection interval: configurable in code (self.vlm_interval)
        - For RTSP live streams, processing continues until stopped
        """)

    return demo


if __name__ == "__main__":
    os.makedirs("./configs", exist_ok=True)
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./uploaded_videos", exist_ok=True)

    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=9007, share=False)
