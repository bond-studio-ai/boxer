"""GCS replay mailbox with optional Pub/Sub delivery for Boxer workers."""

import json
import logging
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gcp_worker.contract import validate

log = logging.getLogger(__name__)


def _run_child(command, timeout):
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return_code = child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def process(blob, bucket, worker_timeout):
    from google.api_core.exceptions import PreconditionFailed

    request = json.loads(blob.download_as_bytes(timeout=30))
    job = validate(request)
    if blob.name != f"requests/{job}.json":
        raise ValueError("Request object name does not match job")
    result_blob = bucket.blob(f"results/{job}.json")
    if result_blob.exists(timeout=30):
        blob.delete(if_generation_match=blob.generation, timeout=30)
        return "already-complete"
    try:
        bucket.blob(f"claims/{job}.json").upload_from_string(
            json.dumps({"claimedAt": time.time()}), if_generation_match=0, timeout=30
        )
    except PreconditionFailed:
        return "already-claimed"

    result = {"version": 1, "jobId": job, "status": "failed", "error": "worker_failed"}
    try:
        remaining = min(worker_timeout, request["expiresAt"] - time.time())
        if remaining <= 0:
            result["error"] = "deadline_exceeded_before_start"
        else:
            with tempfile.TemporaryDirectory(prefix="boxer-replay-") as directory:
                scratch = Path(directory)
                request_path = scratch / "request.json"
                request_path.write_text(json.dumps(request))
                _run_child(
                    [
                        sys.executable,
                        "-m",
                        "gcp_worker.execute",
                        str(request_path),
                        str(scratch),
                        str(remaining),
                    ],
                    remaining,
                )
                result = json.loads((scratch / "result.json").read_text())
    except subprocess.TimeoutExpired:
        result["error"] = "deadline_exceeded"
    except Exception:
        log.exception("Boxer replay failed: %s", job)
    result_blob.upload_from_string(
        json.dumps(result),
        content_type="application/json",
        if_generation_match=0,
        timeout=30,
    )
    blob.delete(if_generation_match=blob.generation, timeout=30)
    return "complete"


def _notification_blob(message, bucket):
    payload = json.loads(message.data)
    if not isinstance(payload, dict):
        raise ValueError("Notification payload must be an object")
    name = payload.get("name")
    if payload.get("bucket") != bucket.name:
        raise ValueError("Notification bucket does not match")
    if not isinstance(name, str) or not re.fullmatch(
        r"requests/[a-f0-9]{32}\.json", name
    ):
        raise ValueError("Notification is not a Boxer request")
    generation = payload.get("generation")
    if not isinstance(generation, str) or not generation.isdecimal():
        raise ValueError("Notification has no valid generation")
    return bucket.blob(name, generation=int(generation))


def handle_notification(message, bucket, worker_timeout):
    from google.api_core.exceptions import NotFound

    try:
        disposition = process(
            _notification_blob(message, bucket), bucket, worker_timeout
        )
    except NotFound:
        message.ack()
        return
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        log.exception("Discarding invalid Boxer notification: %s", message.message_id)
        message.ack()
        return
    except Exception:
        log.exception("Boxer notification failed and will be retried")
        message.nack()
        return
    if disposition == "already-claimed":
        log.warning("Boxer request is already claimed")
    message.ack()


def run_subscription(bucket, worker_timeout, subscription):
    from google.cloud import pubsub_v1

    with pubsub_v1.SubscriberClient() as subscriber:
        project = os.environ["GOOGLE_CLOUD_PROJECT"]
        path = (
            subscription
            if subscription.startswith("projects/")
            else subscriber.subscription_path(project, subscription)
        )
        future = subscriber.subscribe(
            path,
            lambda message: handle_notification(message, bucket, worker_timeout),
            flow_control=pubsub_v1.types.FlowControl(
                max_messages=1, max_lease_duration=worker_timeout + 300
            ),
            await_callbacks_on_shutdown=True,
        )
        log.info("Boxer Pub/Sub worker ready: %s", path)
        future.result()


def main():
    import torch
    from google.cloud import storage

    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required; CPU fallback is disabled")
    torch.empty(1, device="cuda").sum().item()
    bucket = storage.Client().bucket(os.environ["BOXER_REPLAY_BUCKET"])
    timeout = int(os.environ.get("BOXER_JOB_TIMEOUT_SECONDS", "1800"))
    if not 1 <= timeout <= 3600:
        raise ValueError("Boxer job timeout must be 1..3600 seconds")
    subscription = os.environ.get("BOXER_REPLAY_SUBSCRIPTION")
    if subscription:
        run_subscription(bucket, timeout, subscription)
        return
    while True:
        try:
            for blob in bucket.list_blobs(prefix="requests/", timeout=30):
                process(blob, bucket, timeout)
        except Exception:
            log.exception("Boxer mailbox poll failed")
        time.sleep(5)


if __name__ == "__main__":
    main()
