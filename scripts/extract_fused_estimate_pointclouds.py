#!/usr/bin/env python3
"""Extract aligned-cloud subsets for fused OBB estimates."""

import argparse
import csv
import glob
import os

import numpy as np

from extract_best_box_pointclouds import (
    quaternion_wxyz_to_matrix,
    read_ply_layout,
    write_ply,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--fused-filename",
        default="boxer_3dbbs_fused_min2.csv",
        help="Per-sequence fused-box CSV to extract",
    )
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    fused_csvs = sorted(
        glob.glob(os.path.join(args.results_root, "20260915_12*", args.fused_filename))
    )
    manifest = []
    for fused_csv in fused_csvs:
        sequence = os.path.basename(os.path.dirname(fused_csv))
        boxes = []
        label_counts = {}
        with open(fused_csv, newline="") as source:
            for row in csv.DictReader(source):
                label = row["name"]
                label_counts[label] = label_counts.get(label, 0) + 1
                row["estimate_index"] = label_counts[label]
                row["center"] = np.array(
                    [
                        row["tx_world_object"],
                        row["ty_world_object"],
                        row["tz_world_object"],
                    ],
                    dtype=np.float64,
                )
                row["half_size"] = 0.5 * np.array(
                    [row["scale_x"], row["scale_y"], row["scale_z"]],
                    dtype=np.float64,
                )
                row["rotation"] = quaternion_wxyz_to_matrix(
                    [
                        row["qw_world_object"],
                        row["qx_world_object"],
                        row["qy_world_object"],
                        row["qz_world_object"],
                    ]
                )
                boxes.append(row)

        aligned_path = os.path.join(args.source_root, sequence, "aligned.ply")
        header, offset, vertex_count, dtype = read_ply_layout(aligned_path)
        vertices = np.memmap(
            aligned_path,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=(vertex_count,),
        )
        selected = [[] for _ in boxes]
        combined_chunks = []
        for start in range(0, vertex_count, args.chunk_size):
            chunk = vertices[start : min(start + args.chunk_size, vertex_count)]
            xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"]))
            finite = np.isfinite(xyz).all(axis=1)
            combined_mask = np.zeros(len(chunk), dtype=bool)
            for index, box in enumerate(boxes):
                local = (xyz - box["center"]) @ box["rotation"]
                inside = finite & np.all(
                    np.abs(local) <= box["half_size"] + 1e-7,
                    axis=1,
                )
                if inside.any():
                    selected[index].append(np.array(chunk[inside]))
                    combined_mask |= inside
            if combined_mask.any():
                combined_chunks.append(np.array(chunk[combined_mask]))

        sequence_output = os.path.join(args.output_root, sequence)
        os.makedirs(sequence_output, exist_ok=True)
        for box, chunks in zip(boxes, selected):
            subset = np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
            stem = f"{box['name']}_fused_{box['estimate_index']:02d}"
            output_path = os.path.join(sequence_output, f"{stem}.ply")
            write_ply(output_path, header, subset)
            manifest.append(
                {
                    "sequence": sequence,
                    "estimate": stem,
                    "object": box["name"],
                    "confidence": box["prob"],
                    "points": len(subset),
                    "source_points": vertex_count,
                    "output_ply": os.path.abspath(output_path),
                    "fused_box_csv": os.path.abspath(fused_csv),
                }
            )
            print(f"{sequence} {stem}: {len(subset):,} points")

        combined = (
            np.concatenate(combined_chunks)
            if combined_chunks
            else np.empty(0, dtype=dtype)
        )
        write_ply(os.path.join(sequence_output, "all_fused_estimates.ply"), header, combined)

    os.makedirs(args.output_root, exist_ok=True)
    manifest_path = os.path.join(args.output_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved {len(manifest)} fused-estimate PLYs and manifest {manifest_path}")


if __name__ == "__main__":
    main()
