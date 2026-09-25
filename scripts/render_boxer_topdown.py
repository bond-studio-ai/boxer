#!/usr/bin/env python3
"""Render fused Boxer OBBs and their point subsets over an aligned PLY."""

import argparse
import csv
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extract_best_box_pointclouds import read_ply_layout


COLORS = {
    "tub": (255, 171, 0),
    "vanity": (0, 200, 255),
    "toilet": (255, 92, 92),
    "shower": (84, 255, 159),
    "shower fixture": (210, 105, 255),
    "wall": (235, 235, 235),
    "window": (65, 220, 255),
    "door": (255, 90, 210),
    "soffit": (170, 140, 255),
}
DEFAULT_COLOR = (255, 255, 0)


class TopDownProjection:
    def __init__(self, xmin, ymax, scale, width, height):
        self.xmin = xmin
        self.ymax = ymax
        self.scale = scale
        self.width = width
        self.height = height

    def world_to_image(self, x, y):
        return (
            int(round((x - self.xmin) * self.scale)),
            int(round((self.ymax - y) * self.scale)),
        )


def load_ply_xy_rgb(path):
    _, offset, count, dtype = read_ply_layout(str(path))
    vertices = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(count,))
    xy = np.column_stack((vertices["x"], vertices["y"])).astype(np.float64)
    names = set(dtype.names or ())
    colors = None
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            (vertices["red"], vertices["green"], vertices["blue"])
        ).astype(np.float64)
        if colors.max(initial=0) > 1.0:
            colors /= 255.0
    return xy, colors


def build_scan_projection(ply_xy, layout_xy, colors=None, longest_side=1600):
    bounds = np.vstack((ply_xy, layout_xy)) if len(layout_xy) else ply_xy
    xmin, ymin = bounds.min(axis=0) - 0.15
    xmax, ymax = bounds.max(axis=0) + 0.15
    scale = (longest_side - 1) / max(xmax - xmin, ymax - ymin)
    width = int(np.ceil((xmax - xmin) * scale)) + 1
    height = int(np.ceil((ymax - ymin) * scale)) + 1
    xs = ((ply_xy[:, 0] - xmin) * scale).astype(np.int32)
    ys = ((ymax - ply_xy[:, 1]) * scale).astype(np.int32)
    valid = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
    counts = np.zeros((height, width), dtype=np.float64)
    np.add.at(counts, (ys[valid], xs[valid]), 1.0)
    if colors is not None:
        sums = np.zeros((height, width, 3), dtype=np.float64)
        for channel in range(3):
            np.add.at(sums[:, :, channel], (ys[valid], xs[valid]), colors[valid, channel])
        rgb = np.clip(sums / np.maximum(counts, 1)[:, :, None] * 255, 0, 255).astype(np.uint8)
    else:
        gray = np.log1p(counts)
        gray = (gray / max(float(gray.max()), 1e-12) * 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    return TopDownProjection(xmin, ymax, scale, width, height), rgb


def try_font(size=18):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_polygon(draw, projection, corners, color, width=4):
    pixels = [projection.world_to_image(x, y) for x, y in corners]
    draw.line(pixels + [pixels[0]], fill=color, width=width)
    return tuple(np.mean(np.asarray(pixels), axis=0))


def draw_label(draw, text, anchor, font):
    draw.text(
        anchor,
        text,
        fill=(255, 255, 255),
        font=font,
        anchor="mm",
        stroke_width=3,
        stroke_fill=(0, 0, 0),
    )


def quaternion_wxyz_to_matrix(values):
    w, x, y, z = (float(value) for value in values)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def box_xy_hull(row):
    center = np.array(
        [row["tx_world_object"], row["ty_world_object"], row["tz_world_object"]],
        dtype=np.float64,
    )
    half = 0.5 * np.array(
        [row["scale_x"], row["scale_y"], row["scale_z"]], dtype=np.float64
    )
    rotation = quaternion_wxyz_to_matrix(
        [
            row["qw_world_object"],
            row["qx_world_object"],
            row["qy_world_object"],
            row["qz_world_object"],
        ]
    )
    local = np.array(list(product((-1.0, 1.0), repeat=3))) * half
    world_xy = (local @ rotation.T + center)[:, :2].astype(np.float32)
    hull = cv2.convexHull(world_xy).reshape(-1, 2)
    return [tuple(map(float, point)) for point in hull]


def load_boxes(path):
    boxes = []
    counts = {}
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            name = row["name"]
            counts[name] = counts.get(name, 0) + 1
            row["estimate"] = f"{name}_fused_{counts[name]:02d}"
            boxes.append(row)
    return boxes


def overlay_subset(rgb, projection, subset_path, color):
    points, _ = load_ply_xy_rgb(subset_path)
    if not len(points):
        return
    xs = ((points[:, 0] - projection.xmin) * projection.scale).astype(np.int32)
    ys = ((projection.ymax - points[:, 1]) * projection.scale).astype(np.int32)
    valid = (xs >= 0) & (ys >= 0) & (xs < rgb.shape[1]) & (ys < rgb.shape[0])
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[ys[valid], xs[valid]] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    selected = mask > 0
    rgb[selected] = (
        0.35 * rgb[selected].astype(np.float32)
        + 0.65 * np.asarray(color, dtype=np.float32)
    ).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-ply", required=True)
    parser.add_argument("--fused-csv", required=True)
    parser.add_argument("--subset-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--longest-side", type=int, default=1600)
    args = parser.parse_args()

    boxes = load_boxes(args.fused_csv)
    ply_xy, base_colors = load_ply_xy_rgb(Path(args.aligned_ply))
    layout = np.vstack(
        [np.asarray(box_xy_hull(box), dtype=np.float64) for box in boxes]
    ) if boxes else np.zeros((0, 2), dtype=np.float64)
    projection, rgb = build_scan_projection(
        ply_xy,
        layout,
        colors=base_colors,
        longest_side=args.longest_side,
    )

    for box in boxes:
        color = COLORS.get(box["name"], DEFAULT_COLOR)
        subset = Path(args.subset_dir) / f"{box['estimate']}.ply"
        if subset.is_file():
            overlay_subset(rgb, projection, subset, color)

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = try_font(18)
    for box in boxes:
        color = COLORS.get(box["name"], DEFAULT_COLOR)
        center = draw_polygon(draw, projection, box_xy_hull(box), color, width=4)
        draw_label(
            draw,
            f"{box['name']} {float(box['prob']):.2f}",
            center,
            font,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved {output} with {len(boxes)} fused boxes")


if __name__ == "__main__":
    main()
