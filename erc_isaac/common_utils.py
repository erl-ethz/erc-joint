from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate [w, x, y, z] -> [w, -x, -y, -z]."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

class Recorder:
    """Record Isaac Lab viewport frames to MP4 at 60 fps."""

    def __init__(self, playback_speed: float, mp4_path: str, dt: float, sim, resolution: tuple[int, int]=(1280, 720), camera_eye: list[float] | None=None, camera_target: list[float] | None=None) -> None:
        self.playback_speed = playback_speed
        self.mp4_path = Path(mp4_path)
        self.dt = dt
        self.sim = sim
        self.resolution = resolution
        self.camera_eye = camera_eye if camera_eye is not None else [3.5, 3.5, 3.5]
        self.camera_target = camera_target if camera_target is not None else [0.0, 0.0, 0.0]
        self.target_fps = 60
        self.sim_time_per_video_frame = 1.0 / self.target_fps * playback_speed
        self.current_sim_time = 0.0
        self.video_writer = None
        self.initialized = False
        self.frame_count = 0
        self._rgb_annotator = None

    def record_frame(self) -> None:
        """Record a frame when the playback schedule requires it."""
        expected_frame_count = int(self.current_sim_time / self.sim_time_per_video_frame)
        frames_to_write = expected_frame_count - self.frame_count
        if frames_to_write > 0:
            if not self.initialized:
                self._initialize()
            frame = self._capture_frame()
            if frame is not None and frame.size > 0:
                for _ in range(frames_to_write):
                    self.video_writer.write(frame)
                    self.frame_count += 1
        self.current_sim_time += self.dt

    def _initialize(self) -> None:
        try:
            import cv2
            import omni.replicator.core as rep
            self.mp4_path.parent.mkdir(parents=True, exist_ok=True)
            self.sim.set_camera_view(self.camera_eye, self.camera_target)
            viewport_api = rep.create.render_product('/OmniverseKit_Persp', self.resolution)
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
            self._rgb_annotator.attach([viewport_api])
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(self.mp4_path), fourcc, self.target_fps, self.resolution)
            self.initialized = True
            print(f'[Recorder] Recording {self.resolution[0]}x{self.resolution[1]} @ {self.target_fps} fps -> {self.mp4_path}')
        except Exception as exc:
            import carb
            carb.log_error(f'Failed to initialize recorder: {exc}')
            self.initialized = False

    def _capture_frame(self) -> np.ndarray | None:
        try:
            import cv2
            rgb_data = self._rgb_annotator.get_data()
            if rgb_data is None or rgb_data.size == 0:
                return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            frame_rgb = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            return cv2.cvtColor(frame_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        except Exception as exc:
            import carb
            carb.log_error(f'Frame capture error: {exc}')
            return None

    def save(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            print(f'[Recorder] Video saved to {self.mp4_path} ({self.frame_count} frames)')
            self.video_writer = None