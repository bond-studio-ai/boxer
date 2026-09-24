# Boxer GCP replay

This repository's `feature/integration` work adds the Bond input adapter: a
timestamped MP4, aligned binary PLY, registered poses and ARKit camera
intrinsics. The isolated GCP worker preserves that interface and produces
`spatiallm_bboxes.txt` plus diagnostic CSVs. It does not publish callbacks or
change production routing.

The request is version 1 and uses one random lowercase hexadecimal job ID. Its
four objects must be exactly:

- `inputs/<job>/video.mp4`
- `inputs/<job>/aligned.ply`
- `inputs/<job>/output_poses_registered.txt`
- `inputs/<job>/arkit_poses.json`

Submit `requests/<job>.json` only after all inputs exist. Results are written
under `outputs/<job>/` and the terminal envelope is `results/<job>.json`.
Claims are immutable, so a crashed worker requires inspection before a new job
ID is submitted. The worker accepts Pub/Sub delivery through
`BOXER_REPLAY_SUBSCRIPTION` and otherwise polls the isolated bucket.

The image pins public model revision
`bfba91291fb3f137c18eb6e45362eeda827eac7b` and verifies SHA-256 for all three
checkpoints during the build. No Hugging Face credential is present in the
image or runtime.

The code and model card declare CC BY-NC 4.0. A technical validation deployment
does not establish permission for commercial production use. Obtain the
company's license approval or separate permission from Meta before routing
commercial traffic.
