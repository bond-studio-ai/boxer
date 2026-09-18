# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for a video, timestamped camera poses, and an aligned PLY cloud."""

import json
import os

import cv2
import numpy as np
import torch

from loaders.base_loader import BaseLoader
from utils.tw.obb import ObbTW
from utils.tw.pose import PoseTW


_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _read_binary_ply_xyz(path: str, max_points: int, seed: int = 0) -> np.ndarray:
    """Sample XYZ fields directly from a binary little-endian PLY."""
    properties = []
    vertex_count = None
    with open(path, "rb") as f:
        first = f.readline().decode("ascii").strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        in_vertex = False
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.decode("ascii").strip()
            fields = text.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                pass
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertex = True
            elif fields and fields[0] == "element":
                in_vertex = False
            elif in_vertex and fields and fields[0] == "property":
                if fields[1] == "list":
                    raise ValueError(
                        "List properties are unsupported in the vertex element"
                    )
                properties.append((fields[2], _PLY_DTYPES[fields[1]]))
            elif text == "end_header":
                data_offset = f.tell()
                break

    if vertex_count is None or not {"x", "y", "z"}.issubset(dict(properties)):
        raise ValueError(f"PLY vertex XYZ fields are missing: {path}")

    vertices = np.memmap(
        path,
        dtype=np.dtype(properties),
        mode="r",
        offset=data_offset,
        shape=(vertex_count,),
    )
    count = min(vertex_count, max_points)
    if count == vertex_count:
        indices = np.arange(vertex_count)
    else:
        # One deterministic random sample from each equal-width part of the file.
        rng = np.random.default_rng(seed)
        edges = np.linspace(0, vertex_count, count + 1, dtype=np.int64)
        indices = edges[:-1] + (rng.random(count) * np.diff(edges)).astype(np.int64)
    xyz = np.column_stack(
        (
            vertices["x"][indices],
            vertices["y"][indices],
            vertices["z"][indices],
        )
    )
    xyz = np.asarray(xyz, dtype=np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def _quaternion_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _load_poses_with_nearest_intrinsics(
    pose_path: str,
    arkit_pose_path: str,
    fps: float = 2.0,
    max_time_delta_s: float = 0.05,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Load registered poses and pair them with the nearest ARKit calibration."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if max_time_delta_s < 0.0:
        raise ValueError("max_time_delta_s must be nonnegative")

    pose_rows = np.atleast_2d(
        np.loadtxt(pose_path, comments="#", dtype=np.float64)
    )
    if pose_rows.shape[1] != 8:
        raise ValueError(f"Expected 8 pose columns, found {pose_rows.shape[1]}")

    with open(arkit_pose_path, encoding="utf-8") as source:
        payload = json.load(source)
    samples = payload.get("data", payload) if isinstance(payload, dict) else payload
    samples = list(samples.values()) if isinstance(samples, dict) else list(samples)
    samples = [
        sample
        for sample in samples
        if "timestamp" in sample
        and len(sample.get("cameraIntrinsics", ())) >= 8
        and "cameraResolution" in sample
    ]
    if not samples:
        raise ValueError(f"No calibrated ARKit samples found in {arkit_pose_path}")
    samples.sort(key=lambda sample: float(sample["timestamp"]))
    sample_times = np.asarray(
        [float(sample["timestamp"]) for sample in samples], dtype=np.float64
    )

    pose_times = pose_rows[:, 0] / fps
    right = np.searchsorted(sample_times, pose_times, side="left")
    right = np.clip(right, 0, len(sample_times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(
        np.abs(sample_times[left] - pose_times)
        <= np.abs(sample_times[right] - pose_times),
        left,
        right,
    )
    deltas = np.abs(sample_times[nearest] - pose_times)
    keep = deltas <= max_time_delta_s + 1e-12
    return (
        pose_rows[keep],
        [samples[index] for index in nearest[keep]],
        deltas[keep],
    )


def _intrinsics_for_display_orientation(
    sample: dict, decoded_width: int, decoded_height: int
) -> tuple[float, float, float, float, float, float]:
    """Orient an ARKit calibration like the auto-rotated decoded video."""
    intrinsics = sample["cameraIntrinsics"]
    resolution = sample["cameraResolution"]
    source_width = float(resolution["width"])
    source_height = float(resolution["height"])
    fx = float(intrinsics[0])
    fy = float(intrinsics[4])
    cx = float(intrinsics[6])
    cy = float(intrinsics[7])

    source_landscape = source_width > source_height
    decoded_landscape = decoded_width > decoded_height
    if source_landscape != decoded_landscape:
        fx, fy = fy, fx
        cx, cy = cy, cx
        source_width, source_height = source_height, source_width
    return fx, fy, cx, cy, source_width, source_height


class VideoPlyLoader(BaseLoader):
    """Read frames at pose timestamps and reuse an aligned world point cloud."""

    def __init__(
        self,
        sequence_dir: str,
        skip_frames: int = 1,
        max_frames: int | None = None,
        start_frame: int = 1,
        max_cloud_points: int = 250_000,
        fps: float = 2.0,
        max_intrinsics_delta_s: float = 0.05,
    ):
        self.sequence_dir = os.path.abspath(os.path.expanduser(sequence_dir))
        self.scene_id = os.path.basename(self.sequence_dir.rstrip("/"))
        self.video_path = os.path.join(self.sequence_dir, "video.mp4")
        self.pose_path = os.path.join(self.sequence_dir, "output_poses_registered.txt")
        self.arkit_pose_path = os.path.join(self.sequence_dir, "arkit_poses.json")
        self.cloud_path = os.path.join(self.sequence_dir, "aligned.ply")
        for path in (
            self.video_path,
            self.pose_path,
            self.arkit_pose_path,
            self.cloud_path,
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(path)

        rows, intrinsics_samples, intrinsics_deltas = _load_poses_with_nearest_intrinsics(
            self.pose_path,
            self.arkit_pose_path,
            fps=fps,
            max_time_delta_s=max_intrinsics_delta_s,
        )
        rows = rows[start_frame - 1 :: skip_frames]
        intrinsics_samples = intrinsics_samples[start_frame - 1 :: skip_frames]
        intrinsics_deltas = intrinsics_deltas[start_frame - 1 :: skip_frames]
        if max_frames is not None:
            rows = rows[:max_frames]
            intrinsics_samples = intrinsics_samples[:max_frames]
            intrinsics_deltas = intrinsics_deltas[:max_frames]
        self.rows = rows
        self.intrinsics_samples = intrinsics_samples
        self.intrinsics_deltas = intrinsics_deltas
        self.fps = float(fps)
        self.length = len(rows)
        self.index = 0
        self.resize = None
        self.camera = "rgb"
        self.device_name = "timestamped-video"

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.video_path}")
        # OpenCV applies MP4 display rotation when this property is enabled.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, image = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"Cannot decode video: {self.video_path}")
        self.decoded_height, self.decoded_width = image.shape[:2]

        self.sdp_w = torch.from_numpy(
            _read_binary_ply_xyz(self.cloud_path, max_cloud_points, seed=0)
        )
        print(
            f"VideoPlyLoader: {self.scene_id}, {self.length} timestamps, "
            f"{len(self.sdp_w)} sampled cloud points, "
            f"decoded {self.decoded_width}x{self.decoded_height}, "
            f"maximum intrinsics offset {self.intrinsics_deltas.max(initial=0.0) * 1000:.1f}ms"
        )
        self._init_prefetch()

    def load(self, idx):
        row = self.rows[idx]
        source_frame, x, y, z, qx, qy, qz, qw = row
        timestamp_s = source_frame / self.fps

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_s) * 1000.0)
        ok, image_bgr = cap.read()
        cap.release()
        if not ok:
            raise IOError(
                f"Cannot decode {self.video_path} at timestamp {timestamp_s:.6f}s"
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        fx, fy, cx, cy, calib_width, calib_height = (
            _intrinsics_for_display_orientation(
                self.intrinsics_samples[idx], width, height
            )
        )
        scale_x = width / calib_width
        scale_y = height / calib_height
        fx, fy = fx * scale_x, fy * scale_y
        cx, cy = cx * scale_x, cy * scale_y

        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)):
                resize_h, resize_w = self.resize
            else:
                resize_h = resize_w = self.resize
            fx *= resize_w / width
            cx *= resize_w / width
            fy *= resize_h / height
            cy *= resize_h / height
            image_rgb = cv2.resize(
                image_rgb, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
            )
            height, width = resize_h, resize_w

        rotation = _quaternion_xyzw_to_matrix([qx, qy, qz, qw])
        translation = np.array([x, y, z], dtype=np.float32)
        T_world_camera = PoseTW.from_Rt(
            torch.from_numpy(rotation), torch.from_numpy(translation)
        )
        camera = self.pinhole_from_K(
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            valid_radius=(width, height),
        )

        return {
            "img0": self.img_to_tensor(image_rgb),
            "cam0": camera.float(),
            "T_world_rig0": T_world_camera.float(),
            "sdp_w": self.sdp_w,
            "time_ns0": int(round(timestamp_s * 1e9)),
            "obbs": ObbTW(torch.zeros(0, 165)),
            "bb2d0": torch.zeros(0, 4, dtype=torch.float32),
            "gt_labels": [],
        }
