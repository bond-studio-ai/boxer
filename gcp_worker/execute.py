"""Download one replay, run the headless Boxer CLI, and upload bounded outputs."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from gcp_worker.contract import DEFAULT_LABELS, INPUT_FILES, validate

OUTPUT_FILES = {
    "spatiallmBboxesPath": ("spatiallm_bboxes.txt", True),
    "boxes3dPath": ("boxer_3dbbs.csv", True),
    "boxes3dFusedPath": ("boxer_3dbbs_fused.csv", False),
    "boxes2dPath": ("owl_2dbbs.csv", False),
}


def _args(request, scratch):
    settings = request.get("settings", {})
    inputs = {key: scratch / filename for key, filename in INPUT_FILES.items()}
    return [
        sys.executable,
        "/app/run_boxer.py",
        "--video",
        str(inputs["video"]),
        "--point_cloud",
        str(inputs["pointCloud"]),
        "--poses_registered",
        str(inputs["posesRegistered"]),
        "--poses_arkit",
        str(inputs["posesArkit"]),
        "--labels=" + ",".join(request.get("labels", DEFAULT_LABELS)),
        "--output_dir",
        str(scratch / "output"),
        "--skip_viz",
        "--video_fps",
        str(settings.get("videoFps", 2.0)),
        "--skip_n",
        str(int(settings.get("skipFrames", 1))),
        "--max_n",
        str(int(settings.get("maxFrames", 99999))),
        "--thresh2d",
        str(settings.get("threshold2d", 0.25)),
        "--thresh3d",
        str(settings.get("threshold3d", 0.5)),
        "--detector_hw",
        str(int(settings.get("detectorSize", 960))),
        "--extent-method",
        settings.get("extentMethod", "mean"),
        "--envelope-padding-m",
        str(settings.get("envelopePaddingM", 0.0)),
    ]


def execute(request, bucket, scratch, timeout, gpu_name):
    job = validate(request)
    started = time.monotonic()
    timings = {}
    phase = time.monotonic()
    for key, filename in INPUT_FILES.items():
        bucket.blob(request["inputs"][key]).download_to_filename(
            str(scratch / filename), timeout=300
        )
    timings["inputDownloadSeconds"] = time.monotonic() - phase

    output = scratch / "output"
    output.mkdir()
    phase = time.monotonic()
    subprocess.run(_args(request, scratch), check=True, timeout=timeout)
    timings["inferenceSeconds"] = time.monotonic() - phase

    result = {
        "version": 1,
        "jobId": job,
        "status": "succeeded",
        "gpu": gpu_name,
    }
    phase = time.monotonic()
    for key, (filename, required) in OUTPUT_FILES.items():
        path = output / filename
        if not path.is_file():
            if required:
                raise ValueError(f"Missing required Boxer artifact: {filename}")
            result[key] = None
            continue
        remote = f"outputs/{job}/{filename}"
        bucket.blob(remote).upload_from_filename(
            str(path), if_generation_match=0, timeout=120
        )
        result[key] = remote
    timings["artifactUploadSeconds"] = time.monotonic() - phase
    result["timings"] = timings
    result["elapsedSeconds"] = time.monotonic() - started
    return result


def child_main(request_path, scratch_path, timeout):
    import torch
    from google.cloud import storage

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; CPU fallback is disabled")
    request = json.loads(Path(request_path).read_text())
    bucket = storage.Client().bucket(os.environ["BOXER_REPLAY_BUCKET"])
    result = execute(
        request,
        bucket,
        Path(scratch_path),
        timeout,
        torch.cuda.get_device_name(0),
    )
    (Path(scratch_path) / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    child_main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
