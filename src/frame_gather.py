
import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
import yaml

from utils.config import load_config


class FrameSynchronizer:
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        cam_defs = cfg.get("sources", [])
        if not cam_defs:
            raise ValueError("No sources defined in config['sources']")

        # Extract IDs and URLs
        self.camera_ids = [int(cam["id"]) for cam in cam_defs]
        self.sources = [cam["url"] for cam in cam_defs]
        self.camera_status = {cam_id: False for cam_id in self.camera_ids}

        # how many consecutive read failures before declaring “down”
        self.max_failures = cfg.get("max_failures", 5)
        self.fail_counts = {cam_id: 0 for cam_id in self.camera_ids}

        # track last successful frame time for timeout detection
        self.last_success_time = {cam_id: 0.0 for cam_id in self.camera_ids}
        # seconds without a new frame before marking down
        self.disconnect_timeout = cfg.get("disconnect_timeout", 5)
        self.report_url = cfg.get("dashboard_alarm", None)
        self.buffer_size = cfg.get("buffer_size", 10)
        self.output_dir = cfg.get("output_dir", "")
        self.save_tiles = bool(self.output_dir)
        if self.save_tiles:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logging
        log_level = getattr(
            logging, cfg.get("log_level", "ERROR"), logging.ERROR
        )
        logging.basicConfig(
            level=log_level, format="%(asctime)s [%(levelname)s] %(message)s"
        )
        self.logger = logging.getLogger("FrameGather")

        # Create frame buffers
        self.frame_buffers = {
            cam_id: deque(maxlen=self.buffer_size) for cam_id in self.camera_ids
        }

        self.threads = []
        self.sync_thread = None
        self._stop_event = threading.Event()
        self._save_counter = 0

    def start(self):
        # Start capture threads
        for cam_id, url in zip(self.camera_ids, self.sources):
            t = threading.Thread(
                target=self._capture_loop, args=(cam_id, url), daemon=True
            )
            t.start()
            self.threads.append(t)
            self.logger.info(f"Started capture thread for camera {cam_id}")

        # Start synchronization thread
        self.sync_thread = threading.Thread(
            target=self._gather_synchronized_frames, daemon=True
        )
        self.sync_thread.start()
        self.logger.info("Started frame synchronization thread")

    def stop(self):
        self._stop_event.set()
        self.logger.info("Stopping all threads...")
        for t in self.threads:
            t.join(timeout=1)
        if self.sync_thread:
            self.sync_thread.join(timeout=1)
        self.logger.info("All threads stopped")

    def _capture_loop(self, cam_id: int, url: str):
        cap = None
        is_stream = url.lower().startswith("rtsp://")
        frame_delay = 0.0
        while not self._stop_event.is_set():
            try:
                # (re)connect if needed
                if cap is None or not cap.isOpened():
                    self.logger.info(f"[{cam_id}] Connecting to {url}...")
                    if cap:
                        cap.release()
                    cap = cv2.VideoCapture(url)
                    if cap.isOpened() and not is_stream:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps > 0:
                            frame_delay = 1.0 / fps
                        else:
                            frame_delay = 0.0
                    else:
                        frame_delay = 0.0
                    # allow immediate wake-up on stop
                    if self._stop_event.wait(1):
                        break

                ret, frame = cap.read()
                if not ret:
                    if not is_stream:
                        # Loop video file
                        cap.release()
                        cap = cv2.VideoCapture(url)
                        continue
                    else:
                        self.fail_counts[cam_id] += 1
                        self.logger.error(
                            f"[{cam_id}] Read failed ({self.fail_counts[cam_id]})"
                        )
                        # after N failures, mark down immediately
                        if self.fail_counts[cam_id] >= self.max_failures:
                            self.logger.error(f"[{cam_id}] Marking camera DOWN")
                            self.camera_status[cam_id] = False
                            self.frame_buffers[cam_id].clear()
                            cap.release()
                            cap = None
                            try:
                                # avoid attempting network calls if we're stopping
                                if self._stop_event.is_set():
                                    self.logger.info(
                                        f"[{cam_id}] Stop requested, skipping DOWN report"
                                    )
                                else:
                                    if self.report_url:
                                        payload = {
                                            "camera_id": cam_id,
                                            "timestamp": time.time(),
                                            "message": f"No new frames for {self.max_failures} try, marking DOWN",
                                        }
                                        resp = requests.post(
                                            self.report_url, json=payload, timeout=5
                                        )
                                        resp.raise_for_status()
                                        self.logger.info(
                                            f"Reported DOWN for {cam_id} to {self.report_url}"
                                        )
                            except Exception:
                                # we log this but don’t raise, so we don’t break your gather loop
                                self.logger.exception(
                                    f"Failed to POST report for camera {cam_id}"
                                )
                            # wait but wake immediately if stop requested
                            if self._stop_event.wait(30):
                                break
                    continue

                # Good frame
                self.fail_counts[cam_id] = 0
                now = time.time()
                self.last_success_time[cam_id] = now
                if not self.camera_status[cam_id]:
                    self.logger.info(f"[{cam_id}] Camera UP again")
                self.camera_status[cam_id] = True

                # Buffer frame
                self.frame_buffers[cam_id].append((frame, now))

                if frame_delay > 0:
                    time.sleep(frame_delay)

            except Exception as e:
                self.logger.exception(f"[{cam_id}] Capture exception: {e}")
                self.camera_status[cam_id] = False
                if cap:
                    cap.release()
                cap = None
                # wait but allow quick exit on stop
                if self._stop_event.wait(5):
                    break

    def _gather_synchronized_frames(self):
        while not self._stop_event.is_set():
            try:
                now = time.time()
                # detect stale cameras
                for cam_id in self.camera_ids:
                    if (
                        self.camera_status[cam_id]
                        and now - self.last_success_time[cam_id]
                        > self.disconnect_timeout
                    ):
                        self.logger.error(
                            f"[{cam_id}] No new frames for {self.disconnect_timeout}s, marking DOWN"
                        )
                        self.camera_status[cam_id] = False
                        self.frame_buffers[cam_id].clear()
                        try:
                            # avoid attempting network calls if we're stopping
                            if self._stop_event.is_set():
                                self.logger.info(
                                    f"[{cam_id}] Stop requested, skipping DOWN report"
                                )
                            else:
                                if self.report_url:
                                    payload = {
                                        "camera_id": cam_id,
                                        "timestamp": time.time(),
                                        "message": f"No new frames for {self.disconnect_timeout}s, marking DOWN",
                                    }
                                    resp = requests.post(
                                        self.report_url, json=payload, timeout=5
                                    )
                                    resp.raise_for_status()
                                    self.logger.info(
                                        f"Reported DOWN for {cam_id} to {self.report_url}"
                                    )
                        except Exception:
                            # we log this but don’t raise, so we don’t break your gather loop
                            self.logger.exception(
                                f"Failed to POST report for camera {cam_id}"
                            )

                result = self._sync_once()
                if result:
                    frames, _ = result
                    if self.save_tiles:
                        # derive frame size from first valid entry
                        h, w = frames[0][0].shape[:2]
                        self._save_counter += 1
                        self.save_tiled_frames(frames, self._save_counter, h, w)

                # allow immediate wake-up on stop
                if self._stop_event.wait(0.01):
                    break
            except Exception as e:
                self.logger.exception(f"Error during synchronization: {e}")

    def _sync_once(self):
        valid = [
            cid
            for cid in self.camera_ids
            if self.camera_status[cid] and self.frame_buffers[cid]
        ]
        if not valid:
            return None

        target = min(self.frame_buffers[cid][-1][1] for cid in valid)
        synced = []
        # choose closest frame by timestamp
        for cam_id in self.camera_ids:
            buf = self.frame_buffers[cam_id]
            if self.camera_status[cam_id] and buf:
                frame_ts = min(buf, key=lambda x: abs(x[1] - target))
                synced.append((frame_ts[0], cam_id))
                # drop older
                while buf and buf[0][1] < target:
                    buf.popleft()
            else:
                synced.append((None, cam_id))

        # Determine shape from first valid frame
        shape_frame = next((f for f, cid in synced if f is not None), None)
        h, w = shape_frame.shape[:2] if shape_frame is not None else (720, 1280)

        # Replace None with black frames
        synced = [(np.zeros((h, w, 3), dtype=np.uint8) if f is None else f, cid) for f, cid in synced]

        return synced, target

    def save_tiled_frames(self, frames_list, save_num: int, fh: int, fw: int):
        n = len(frames_list)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        canvas = np.zeros((rows * fh, cols * fw, 3), dtype=np.uint8)
        for idx, (frame, cam_id) in enumerate(frames_list):
            r, c = divmod(idx, cols)
            canvas[r * fh : (r + 1) * fh, c * fw : (c + 1) * fw] = frame
        path = os.path.join(self.output_dir, f"synced_{save_num}.jpg")
        cv2.imwrite(path, canvas)
        self.logger.info(f"Saved tiled image: {path}")

    def get_synchronized(self):
        while not self._stop_event.is_set():
            res = self._sync_once()
            if res:
                return res
            # allow immediate wake-up on stop
            if self._stop_event.wait(0.01):
                break
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Multi-source frame synchronizer")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to YAML config",
    )
    args = parser.parse_args()

    sync = FrameSynchronizer(args.config)
    sync.start()
    try:
        while True:
            data = sync.get_synchronized()
            if data:
                frames, num = data
                for frame, cam in frames:
                    cv2.imshow(f"Cam {cam}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        sync.stop()
        cv2.destroyAllWindows()