#!/usr/bin/env python3
"""Extract RGB point-cloud subsets inside the best oriented 3D boxes."""

import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np


PLY_DTYPES = {
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


def read_ply_layout(path):
    properties = []
    vertex_count = None
    header_lines = []
    with open(path, "rb") as source:
        in_vertices = False
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            header_lines.append(line)
            text = line.decode("ascii").strip()
            fields = text.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                pass
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertices = True
            elif fields and fields[0] == "element":
                in_vertices = False
            elif in_vertices and fields and fields[0] == "property":
                if fields[1] == "list":
                    raise ValueError("List properties in vertices are unsupported")
                properties.append((fields[2], PLY_DTYPES[fields[1]]))
            elif text == "end_header":
                offset = source.tell()
                break
    if vertex_count is None or not {"x", "y", "z"}.issubset(dict(properties)):
        raise ValueError(f"PLY vertex XYZ fields are missing: {path}")
    return b"".join(header_lines), offset, vertex_count, np.dtype(properties)


def quaternion_wxyz_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        raise ValueError("Zero-length quaternion")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def write_ply(path, original_header, vertices):
    header = re.sub(
        rb"(?m)^element vertex \d+$",
        f"element vertex {len(vertices)}".encode("ascii"),
        original_header,
    )
    with open(path, "wb") as target:
        target.write(header)
        vertices.tofile(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    args = parser.parse_args()

    boxes_by_sequence = defaultdict(list)
    with open(args.boxes, newline="") as source:
        for row in csv.DictReader(source):
            row["center"] = np.array(
                [row["center_x"], row["center_y"], row["center_z"]],
                dtype=np.float64,
            )
            row["half_size"] = 0.5 * np.array(
                [row["size_x"], row["size_y"], row["size_z"]],
                dtype=np.float64,
            )
            row["rotation"] = quaternion_wxyz_to_matrix(
                [row["qw"], row["qx"], row["qy"], row["qz"]]
            )
            boxes_by_sequence[row["sequence"]].append(row)

    manifest = []
    os.makedirs(args.output_root, exist_ok=True)
    for sequence, boxes in sorted(boxes_by_sequence.items()):
        source_path = os.path.join(args.source_root, sequence, "aligned.ply")
        header, offset, vertex_count, dtype = read_ply_layout(source_path)
        vertices = np.memmap(
            source_path,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=(vertex_count,),
        )
        selected_chunks = {box["object"]: [] for box in boxes}
        combined_chunks = []
        for start in range(0, vertex_count, args.chunk_size):
            chunk = vertices[start : min(start + args.chunk_size, vertex_count)]
            xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"]))
            finite = np.isfinite(xyz).all(axis=1)
            combined_mask = np.zeros(len(chunk), dtype=bool)
            for box in boxes:
                local = (xyz - box["center"]) @ box["rotation"]
                inside = finite & np.all(
                    np.abs(local) <= box["half_size"] + 1e-7,
                    axis=1,
                )
                if inside.any():
                    selected_chunks[box["object"]].append(np.array(chunk[inside]))
                    combined_mask |= inside
            if combined_mask.any():
                combined_chunks.append(np.array(chunk[combined_mask]))

        sequence_output = os.path.join(args.output_root, sequence)
        os.makedirs(sequence_output, exist_ok=True)
        for box in boxes:
            object_name = box["object"]
            chunks = selected_chunks[object_name]
            subset = np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
            output_path = os.path.join(sequence_output, f"{object_name}.ply")
            write_ply(output_path, header, subset)
            manifest.append(
                {
                    "sequence": sequence,
                    "object": object_name,
                    "points": len(subset),
                    "source_points": vertex_count,
                    "fraction": f"{len(subset) / vertex_count:.8f}",
                    "output_ply": os.path.abspath(output_path),
                }
            )
            print(f"{sequence} {object_name}: {len(subset):,} points")

        combined = (
            np.concatenate(combined_chunks)
            if combined_chunks
            else np.empty(0, dtype=dtype)
        )
        combined_path = os.path.join(sequence_output, "all_best_boxes.ply")
        write_ply(combined_path, header, combined)
        print(f"{sequence} combined: {len(combined):,} unique points")

    manifest_path = os.path.join(args.output_root, "manifest.csv")
    with open(manifest_path, "w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
