#!/usr/bin/env python3
"""Create tabular and visual summaries for timestamped video/PLY runs."""

import argparse
import csv
import glob
import os
from collections import Counter

from PIL import Image, ImageDraw


LABELS = ("tub", "vanity", "toilet")
COLORS = {
    "tub": "#31b7ff",
    "vanity": "#39d98a",
    "toilet": "#ff9f43",
    "shower": "#b084ff",
    "shower fixture": "#ff66b3",
}


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _best(rows, label):
    matches = [row for row in rows if row["name"] == label]
    return max(matches, key=lambda row: float(row["prob"])) if matches else None


def _caption(image, text, height=34):
    out = Image.new("RGB", (image.width, image.height + height), "#16191d")
    out.paste(image, (0, height))
    ImageDraw.Draw(out).text((8, 9), text, fill="white")
    return out


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root")
    parser.add_argument(
        "--labels",
        default=",".join(LABELS),
        help="Comma-separated labels to include",
    )
    parser.add_argument(
        "--sequence-glob",
        default="20260915_12*",
        help="Glob selecting sequence directories under output_root",
    )
    args = parser.parse_args()
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())

    sequence_dirs = sorted(glob.glob(os.path.join(args.output_root, args.sequence_glob)))
    summary_dir = os.path.join(args.output_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    best_2d_rows = []
    best_3d_rows = []
    count_rows = []
    sequence_data = []
    for sequence_dir in sequence_dirs:
        sequence = os.path.basename(sequence_dir)
        rows_2d = _read_csv(os.path.join(sequence_dir, "owl_2dbbs.csv"))
        rows_3d = _read_csv(os.path.join(sequence_dir, "boxer_3dbbs.csv"))
        counts_2d = Counter(row["name"] for row in rows_2d)
        counts_3d = Counter(row["name"] for row in rows_3d)
        best_2d = {label: _best(rows_2d, label) for label in labels}
        best_3d = {label: _best(rows_3d, label) for label in labels}
        sequence_data.append((sequence, sequence_dir, best_2d))

        for label in labels:
            row_2d = best_2d[label]
            if row_2d is not None:
                best_2d_rows.append(
                    {
                        "sequence": sequence,
                        "object": label,
                        "timestamp_s": f"{int(row_2d['time_ns']) / 1e9:.3f}",
                        "sampled_frame": row_2d["frame_id"],
                        "confidence": row_2d["prob"],
                        "x1": row_2d["x1"],
                        "y1": row_2d["y1"],
                        "x2": row_2d["x2"],
                        "y2": row_2d["y2"],
                        "image_width": row_2d["img_width"],
                        "image_height": row_2d["img_height"],
                    }
                )
            row_3d = best_3d[label]
            if row_3d is not None:
                best_3d_rows.append(
                    {
                        "sequence": sequence,
                        "object": label,
                        "timestamp_s": f"{int(row_3d['time_ns']) / 1e9:.3f}",
                        "confidence": row_3d["prob"],
                        "center_x": row_3d["tx_world_object"],
                        "center_y": row_3d["ty_world_object"],
                        "center_z": row_3d["tz_world_object"],
                        "size_x": row_3d["scale_x"],
                        "size_y": row_3d["scale_y"],
                        "size_z": row_3d["scale_z"],
                        "qw": row_3d["qw_world_object"],
                        "qx": row_3d["qx_world_object"],
                        "qy": row_3d["qy_world_object"],
                        "qz": row_3d["qz_world_object"],
                    }
                )
            count_rows.append(
                {
                    "sequence": sequence,
                    "object": label,
                    "detections_2d": counts_2d[label],
                    "lifted_boxes_3d": counts_3d[label],
                }
            )

    _write_csv(
        os.path.join(summary_dir, "best_2d_boxes.csv"),
        best_2d_rows,
        list(best_2d_rows[0]),
    )
    _write_csv(
        os.path.join(summary_dir, "best_3d_boxes.csv"),
        best_3d_rows,
        list(best_3d_rows[0]),
    )
    _write_csv(
        os.path.join(summary_dir, "detection_counts.csv"),
        count_rows,
        list(count_rows[0]),
    )

    tile_size = (480, 240)
    grid = Image.new(
        "RGB", (tile_size[0] * len(labels), (tile_size[1] + 34) * len(sequence_data))
    )
    for row_index, (sequence, sequence_dir, best_2d) in enumerate(sequence_data):
        for column_index, label in enumerate(labels):
            detection = best_2d[label]
            if detection is None:
                image = Image.new("RGB", tile_size, "#24282d")
                title = f"{sequence[9:15]}  {label}  no detection"
            else:
                frame_id = int(detection["frame_id"])
                path = os.path.join(
                    sequence_dir, "boxer_viz", f"boxer_viz_{frame_id:05d}.jpg"
                )
                image = Image.open(path).convert("RGB").resize(tile_size)
                title = (
                    f"{sequence[9:15]}  {label}  "
                    f"p={float(detection['prob']):.2f}  "
                    f"t={int(detection['time_ns']) / 1e9:.1f}s"
                )
            tile = _caption(image, title)
            grid.paste(
                tile,
                (column_index * tile_size[0], row_index * (tile_size[1] + 34)),
            )
    grid.save(os.path.join(summary_dir, "relevant_frames.jpg"), quality=92)

    for label in labels:
        tiles = []
        for sequence, sequence_dir, best_2d in sequence_data:
            detection = best_2d[label]
            if detection is None:
                continue
            frame_id = int(detection["frame_id"])
            path = os.path.join(
                sequence_dir, "boxer_viz", f"boxer_viz_{frame_id:05d}.jpg"
            )
            image = Image.open(path).convert("RGB").crop((0, 0, 960, 960))
            x1, y1, x2, y2 = (
                float(detection[key]) for key in ("x1", "y1", "x2", "y2")
            )
            box_w, box_h = x2 - x1, y2 - y1
            pad = max(24.0, 0.2 * max(box_w, box_h))
            crop_box = (
                max(0, int(x1 - pad)),
                max(0, int(y1 - pad)),
                min(960, int(x2 + pad)),
                min(960, int(y2 + pad)),
            )
            crop = image.crop(crop_box).resize((320, 320))
            draw = ImageDraw.Draw(crop)
            sx = 320 / max(1, crop_box[2] - crop_box[0])
            sy = 320 / max(1, crop_box[3] - crop_box[1])
            selected_box = (
                int((x1 - crop_box[0]) * sx),
                int((y1 - crop_box[1]) * sy),
                int((x2 - crop_box[0]) * sx),
                int((y2 - crop_box[1]) * sy),
            )
            draw.rectangle(selected_box, outline=COLORS.get(label, "#ffffff"), width=5)
            tiles.append(
                _caption(
                    crop,
                    f"{sequence[9:15]}  p={float(detection['prob']):.2f}",
                )
            )
        if not tiles:
            continue
        sheet = Image.new("RGB", (320 * len(tiles), 354), "#16191d")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, (index * 320, 0))
        safe_label = label.replace(" ", "_")
        sheet.save(os.path.join(summary_dir, f"{safe_label}_boxes.jpg"), quality=94)

    print(f"Saved summaries to {summary_dir}")


if __name__ == "__main__":
    main()
