"""Strict, publication-free contract for isolated Boxer replays."""

import math
import re

INPUT_FILES = {
    "video": "video.mp4",
    "pointCloud": "aligned.ply",
    "posesRegistered": "output_poses_registered.txt",
    "posesArkit": "arkit_poses.json",
}
DEFAULT_LABELS = [
    "tub",
    "vanity",
    "toilet",
    "shower",
    "shower fixture",
    "wall",
    "window",
    "door",
    "soffit",
]
EXTENT_METHODS = {"mean", "robust_envelope", "consensus_envelope"}


def _number(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _integer(value, name, minimum, maximum):
    value = _number(value, name, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(value)


def validate(request):
    if not isinstance(request, dict) or request.get("version") != 1:
        raise ValueError("Boxer request version must be 1")
    job = request.get("jobId")
    if not isinstance(job, str) or not re.fullmatch(r"[a-f0-9]{32}", job):
        raise ValueError("Boxer jobId must be 32 lowercase hexadecimal characters")

    inputs = request.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(INPUT_FILES):
        raise ValueError("Boxer request must contain the four exact input names")
    for key, filename in INPUT_FILES.items():
        if inputs[key] != f"inputs/{job}/{filename}":
            raise ValueError(f"Boxer {key} path does not belong to the job")

    labels = request.get("labels", DEFAULT_LABELS)
    if not isinstance(labels, list) or not 1 <= len(labels) <= 64:
        raise ValueError("Boxer labels must contain 1 through 64 values")
    for label in labels:
        if not isinstance(label, str) or not re.fullmatch(
            r"[A-Za-z0-9 _-]{1,64}", label
        ):
            raise ValueError("Boxer label contains unsupported characters")

    settings = request.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("Boxer settings must be an object")
    unknown = set(settings) - {
        "videoFps",
        "skipFrames",
        "maxFrames",
        "threshold2d",
        "threshold3d",
        "detectorSize",
        "extentMethod",
        "envelopePaddingM",
    }
    if unknown:
        raise ValueError(f"Unknown Boxer settings: {', '.join(sorted(unknown))}")
    _number(settings.get("videoFps", 2.0), "videoFps", 0.1, 120.0)
    _integer(settings.get("skipFrames", 1), "skipFrames", 1, 1000)
    _integer(settings.get("maxFrames", 99999), "maxFrames", 1, 100000)
    _number(settings.get("threshold2d", 0.25), "threshold2d", 0.0, 1.0)
    _number(settings.get("threshold3d", 0.5), "threshold3d", 0.0, 1.0)
    _integer(settings.get("detectorSize", 960), "detectorSize", 64, 4096)
    _number(settings.get("envelopePaddingM", 0.0), "envelopePaddingM", 0.0, 2.0)
    if settings.get("extentMethod", "mean") not in EXTENT_METHODS:
        raise ValueError("Unsupported Boxer extentMethod")

    expires_at = request.get("expiresAt")
    _number(expires_at, "expiresAt", 1, 99999999999)
    return job
