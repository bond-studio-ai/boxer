import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gcp_worker.__main__ import _notification_blob, _run_child
from gcp_worker.contract import INPUT_FILES, validate
from gcp_worker.execute import _args, execute

JOB = "a" * 32


def request():
    return {
        "version": 1,
        "jobId": JOB,
        "inputs": {key: f"inputs/{JOB}/{name}" for key, name in INPUT_FILES.items()},
        "expiresAt": 2000000000,
    }


class ContractTests(unittest.TestCase):
    def test_accepts_bounded_request(self):
        self.assertEqual(validate(request()), JOB)

    def test_rejects_cross_job_input(self):
        value = request()
        value["inputs"]["video"] = f"inputs/{'b' * 32}/video.mp4"
        with self.assertRaisesRegex(ValueError, "does not belong"):
            validate(value)

    def test_rejects_unknown_setting(self):
        value = request()
        value["settings"] = {"callback": "https://example.com"}
        with self.assertRaisesRegex(ValueError, "Unknown"):
            validate(value)

    def test_rejects_fractional_frame_count(self):
        value = request()
        value["settings"] = {"maxFrames": 1.5}
        with self.assertRaisesRegex(ValueError, "integer"):
            validate(value)

    def test_cli_uses_headless_explicit_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(request(), Path(directory))
        self.assertIn("--skip_viz", args)
        self.assertIn("--point_cloud", args)
        self.assertNotIn("--force_cpu", args)

    def test_notification_rejects_non_object_payload(self):
        message = Mock(data=b"[]")
        with self.assertRaisesRegex(ValueError, "must be an object"):
            _notification_blob(message, Mock(name="boxer-replay"))

    @patch("gcp_worker.__main__.os.killpg")
    @patch("gcp_worker.__main__.subprocess.Popen")
    def test_timeout_kills_the_complete_inference_process_group(self, popen, killpg):
        child = popen.return_value
        child.pid = 123
        child.wait.side_effect = [subprocess.TimeoutExpired(["boxer"], 1), 0]
        with self.assertRaises(subprocess.TimeoutExpired):
            _run_child(["boxer"], 1)
        popen.assert_called_once_with(["boxer"], start_new_session=True)
        killpg.assert_called_once()
        self.assertEqual(child.wait.call_count, 2)

    @patch("gcp_worker.execute.subprocess.run")
    def test_uploads_only_declared_outputs(self, run):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)

            def write_outputs(*_args, **_kwargs):
                for filename in ("spatiallm_bboxes.txt", "boxer_3dbbs.csv"):
                    (scratch / "output" / filename).write_text("result")

            run.side_effect = write_outputs
            bucket = Mock()
            bucket.blob.return_value.download_to_filename.side_effect = (
                lambda filename, **_: Path(filename).write_bytes(b"input")
            )
            result = execute(request(), bucket, scratch, 60, "test-gpu")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["gpu"], "test-gpu")
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        uploads = [
            call.args[0]
            for call in bucket.blob.call_args_list
            if call.args[0].startswith("outputs/")
        ]
        self.assertEqual(
            uploads,
            [
                f"outputs/{JOB}/spatiallm_bboxes.txt",
                f"outputs/{JOB}/boxer_3dbbs.csv",
            ],
        )


if __name__ == "__main__":
    unittest.main()
