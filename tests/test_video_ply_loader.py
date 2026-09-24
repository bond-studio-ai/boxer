import json

import numpy as np
import pytest

from loaders.video_ply_loader import (
    VideoPlyLoader,
    _intrinsics_for_display_orientation,
    _load_poses_with_nearest_intrinsics,
)


def test_explicit_loader_paths_must_be_supplied_together():
    with pytest.raises(ValueError, match="must be provided together"):
        VideoPlyLoader(video_path="video.mp4")


def test_load_poses_uses_frame_over_fps_and_drops_distant_samples(tmp_path):
    poses = tmp_path / "output_poses_registered.txt"
    poses.write_text(
        "0 0 0 0 0 0 0 1\n"
        "2 1 0 0 0 0 0 1\n"
        "4 2 0 0 0 0 0 1\n"
    )
    samples = {
        "data": {
            "0.02": {
                "timestamp": 0.02,
                "cameraIntrinsics": [100, 0, 0, 0, 101, 0, 50, 40, 1],
                "cameraResolution": {"width": 100, "height": 80},
            },
            "1.04": {
                "timestamp": 1.04,
                "cameraIntrinsics": [110, 0, 0, 0, 111, 0, 51, 41, 1],
                "cameraResolution": {"width": 100, "height": 80},
            },
            "2.06": {
                "timestamp": 2.06,
                "cameraIntrinsics": [120, 0, 0, 0, 121, 0, 52, 42, 1],
                "cameraResolution": {"width": 100, "height": 80},
            },
        }
    }
    arkit = tmp_path / "arkit_poses.json"
    arkit.write_text(json.dumps(samples))

    rows, matched, deltas = _load_poses_with_nearest_intrinsics(
        str(poses), str(arkit), fps=2.0, max_time_delta_s=0.05
    )

    assert rows[:, 0].tolist() == [0.0, 2.0]
    assert [sample["timestamp"] for sample in matched] == [0.02, 1.04]
    np.testing.assert_allclose(deltas, [0.02, 0.04])


def test_intrinsics_are_oriented_like_portrait_video():
    sample = {
        "cameraIntrinsics": [100, 0, 0, 0, 110, 0, 60, 40, 1],
        "cameraResolution": {"width": 120, "height": 80},
    }
    assert _intrinsics_for_display_orientation(sample, 40, 60) == (
        110.0,
        100.0,
        40.0,
        60.0,
        80.0,
        120.0,
    )
