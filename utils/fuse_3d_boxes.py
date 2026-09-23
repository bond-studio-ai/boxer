#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
3D Bounding Box Fusion System

Fuses a set of ObbTW detections into static, de-duplicated instances.
Uses 3D IoU for matching and confidence-weighted averaging for fusion.

Algorithm Overview:
1. Compute pairwise 3D IoU matrix
2. Cluster detections using connected components (IoU threshold)
3. Fuse boxes within each cluster (confidence-weighted averaging)
4. Filter by minimum detection count threshold
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from utils.file_io import ObbCsvWriter2, read_obb_csv
from utils.tw.obb import ObbTW, iou_mc7, iou_mc7_sparse
from utils.tw.pose import PoseTW, rotation_from_euler
from utils.tw.tensor_utils import (
    pad_string,
    string2tensor,
    tensor2string,
    unpad_string,
)


SPATIALLM_CLASS_ALIASES = {
    "shower": "shower_room",
    "toilet": "toilet",
    "vanity": "sink",
    "tub": "tub",
    "bathtub": "tub",
}

_PLY_DTYPES = {
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


def _spatiallm_class_name(label: str) -> str:
    """Map detector labels to SpatialLM names and normalize other labels."""
    normalized = label.strip().lower().replace("-", " ")
    if normalized in SPATIALLM_CLASS_ALIASES:
        return SPATIALLM_CLASS_ALIASES[normalized]
    return "_".join(normalized.split())


def format_spatiallm_bboxes(obbs: ObbTW) -> list[str]:
    """Format static OBBs as ``bbox_N=Bbox(class,x,y,z,yaw,sx,sy,sz)``."""
    lines = []
    for index, obb in enumerate(obbs):
        label = _spatiallm_class_name(obb.text_string())
        x, y, z = (float(value) for value in obb.T_world_object.t)
        rotation = obb.T_world_object.R
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        extent = obb.bb3_object
        sx = float(extent[1] - extent[0])
        sy = float(extent[3] - extent[2])
        sz = float(extent[5] - extent[4])
        values = ",".join(
            format(value, ".17g") for value in (x, y, z, yaw, sx, sy, sz)
        )
        lines.append(f"bbox_{index}=Bbox({label},{values})")
    return lines


def _read_ply_vertex_layout(path: str) -> tuple[int, int, np.dtype]:
    """Return binary little-endian PLY vertex count, offset, and dtype."""
    properties = []
    vertex_count = None
    is_binary_little_endian = False
    with open(path, "rb") as source:
        if source.readline().decode("ascii").strip() != "ply":
            raise ValueError(f"Not a PLY file: {path}")
        in_vertices = False
        while True:
            line = source.readline()
            if not line:
                raise ValueError(f"PLY header has no end_header: {path}")
            text = line.decode("ascii").strip()
            fields = text.split()
            if fields[:2] == ["format", "binary_little_endian"]:
                is_binary_little_endian = True
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                in_vertices = True
            elif fields and fields[0] == "element":
                in_vertices = False
            elif in_vertices and fields and fields[0] == "property":
                if fields[1] == "list":
                    raise ValueError("List properties in PLY vertices are unsupported")
                properties.append((fields[2], _PLY_DTYPES[fields[1]]))
            elif text == "end_header":
                data_offset = source.tell()
                break
    if not is_binary_little_endian:
        raise ValueError(f"Only binary little-endian PLY files are supported: {path}")
    if vertex_count is None or not {"x", "y", "z"}.issubset(dict(properties)):
        raise ValueError(f"PLY vertex XYZ fields are missing: {path}")
    return vertex_count, data_offset, np.dtype(properties)


def format_spatiallm_point_subset_aabbs(
    obbs: ObbTW, point_cloud_path: str, chunk_size: int = 1_000_000
) -> list[str]:
    """Format world-axis-aligned bounds of points enclosed by each fused OBB."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    count = len(obbs)
    if count == 0:
        return []

    vertex_count, offset, dtype = _read_ply_vertex_layout(point_cloud_path)
    vertices = np.memmap(
        point_cloud_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=(vertex_count,),
    )
    centers = obbs.T_world_object.t.detach().cpu().numpy().astype(np.float64)
    rotations = obbs.T_world_object.R.detach().cpu().numpy().astype(np.float64)
    extents = obbs.bb3_object.detach().cpu().numpy().astype(np.float64)
    lower = extents[:, [0, 2, 4]]
    upper = extents[:, [1, 3, 5]]
    subset_min = np.full((count, 3), np.inf, dtype=np.float64)
    subset_max = np.full((count, 3), -np.inf, dtype=np.float64)
    point_counts = np.zeros(count, dtype=np.int64)

    for start in range(0, vertex_count, chunk_size):
        chunk = vertices[start : min(start + chunk_size, vertex_count)]
        xyz = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(
            np.float64, copy=False
        )
        finite = np.isfinite(xyz).all(axis=1)
        for index in range(count):
            local = (xyz - centers[index]) @ rotations[index]
            inside = finite & np.all(
                (local >= lower[index] - 1e-7)
                & (local <= upper[index] + 1e-7),
                axis=1,
            )
            if inside.any():
                selected = xyz[inside]
                subset_min[index] = np.minimum(
                    subset_min[index], selected.min(axis=0)
                )
                subset_max[index] = np.maximum(
                    subset_max[index], selected.max(axis=0)
                )
                point_counts[index] += int(inside.sum())

    lines = []
    for obb_index, obb in enumerate(obbs):
        if point_counts[obb_index] == 0:
            print(
                "==> Warning: omitting SpatialLM box for "
                f"'{obb.text_string()}': fused point subset is empty"
            )
            continue
        label = _spatiallm_class_name(obb.text_string())
        center = 0.5 * (subset_min[obb_index] + subset_max[obb_index])
        size = subset_max[obb_index] - subset_min[obb_index]
        values = ",".join(
            format(value, ".17g") for value in (*center, 0.0, *size)
        )
        lines.append(f"bbox_{len(lines)}=Bbox({label},{values})")
    return lines


def write_spatiallm_bboxes(
    obbs: ObbTW, output_path: str, point_cloud_path: str | None = None
) -> list[str]:
    """Write and return SpatialLM-style bounding-box declarations."""
    lines = (
        format_spatiallm_point_subset_aabbs(obbs, point_cloud_path)
        if point_cloud_path is not None
        else format_spatiallm_bboxes(obbs)
    )
    with open(output_path, "w", encoding="utf-8") as target:
        if lines:
            target.write("\n".join(lines) + "\n")
    return lines

# =============================================================================
# Shared helper functions (used by both BoundingBox3DFuser and BoundingBox3DTracker)
# =============================================================================


def weighted_yaw_mean(
    angles: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted mean of 1D rotations with 180-degree symmetry (pi-periodic).

    Args:
        angles: (N,) in radians
        weights: (N,) nonnegative

    Returns:
        mean_angle: scalar (radians, in [-pi/2, pi/2) equivalent sense)
        resultant: scalar, magnitude of mean vector (0 = ambiguous)
    """
    angles = angles.float()
    weights = weights.float()

    # Double the angles to resolve pi-periodicity
    phi = 2.0 * angles  # (N,)

    # Weighted sum of unit vectors on the circle
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    x = torch.sum(weights * cos_phi)
    y = torch.sum(weights * sin_phi)

    # Resultant length (can be used as a confidence / concentration)
    resultant = torch.sqrt(x * x + y * y)

    # Handle degenerate case: near-zero resultant => no clear mean direction
    if resultant < eps:
        mean_angle = torch.tensor(0.0, device=angles.device)
    else:
        mean_phi = torch.atan2(y, x)  # in [-pi, pi]
        mean_angle = 0.5 * mean_phi  # back to pi-periodic space

    return mean_angle, resultant


def angular_distance(angle1: float, angle2: float) -> float:
    """
    Compute smallest angular distance between two angles accounting for 180-degree symmetry.

    Args:
        angle1: First angle in radians
        angle2: Second angle in radians

    Returns:
        Distance in radians [0, pi/2]
    """
    # Normalize difference to [-pi, pi]
    diff = (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi

    # Account for 180-degree symmetry: treat theta and theta+pi as equivalent
    # Map to [0, pi/2]
    diff = abs(diff)
    if diff > math.pi / 2:
        diff = math.pi - diff

    return diff


def weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, quantile: float
) -> torch.Tensor:
    """Return a weighted quantile for each column of a 2D tensor."""
    if values.ndim != 2:
        raise ValueError("values must have shape (N, D)")
    if weights.ndim != 1 or weights.shape[0] != values.shape[0]:
        raise ValueError("weights must have shape (N,)")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")

    outputs = []
    normalized_weights = weights / (weights.sum() + 1e-8)
    for dimension in range(values.shape[1]):
        sorted_values, order = torch.sort(values[:, dimension])
        cumulative = torch.cumsum(normalized_weights[order], dim=0)
        index = torch.searchsorted(
            cumulative,
            torch.tensor(quantile, dtype=cumulative.dtype, device=cumulative.device),
        ).clamp(max=len(sorted_values) - 1)
        outputs.append(sorted_values[index])
    return torch.stack(outputs)


def consensus_outer_face(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    choose_lower: bool,
    min_support: int,
    tolerance_m: float,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Estimate an outer face supported by multiple independent frames.

    Multiple detections from one frame count once. Within a frame, the most
    outward face is retained. The outermost face with ``min_support`` nearby
    frame-level observations is returned as their median.
    """
    if values.ndim != 1 or group_ids.ndim != 1 or len(values) != len(group_ids):
        raise ValueError("values and group_ids must be one-dimensional and aligned")

    frame_values = []
    for group_id in torch.unique(group_ids):
        candidates = values[group_ids == group_id]
        frame_values.append(candidates.min() if choose_lower else candidates.max())
    frame_values = torch.stack(frame_values)
    ordered = torch.sort(frame_values, descending=not choose_lower).values
    for candidate in ordered:
        nearby = frame_values[torch.abs(frame_values - candidate) <= tolerance_m]
        if len(nearby) >= min_support:
            return nearby.median()
    return fallback


def align_boxes_r90(
    sizes: torch.Tensor,
    yaw_angles: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Align boxes to canonical orientation accounting for 90-degree rotation symmetry.

    For rectangular boxes, a 90-degree rotation with swapped dimensions represents the same
    object orientation (e.g., 2m x 1m at 0 degrees == 1m x 2m at 90 degrees).

    Strategy:
    1. Find canonical orientation (weighted mean yaw)
    2. For each box, determine if swapping dimensions + rotating 90 degrees gives better alignment
    3. Apply swaps and yaw adjustments where needed

    Args:
        sizes: (M, 3) tensor of [width, height, depth]
        yaw_angles: (M,) tensor of yaw angles in radians
        weights: (M,) tensor of fusion weights

    Returns:
        aligned_sizes: (M, 3) tensor with potentially swapped width/height
        aligned_yaws: (M,) tensor with potentially adjusted yaw angles
    """
    M = len(sizes)
    device = sizes.device

    # Compute reference yaw (weighted mean with 180-degree symmetry)
    ref_yaw, _ = weighted_yaw_mean(yaw_angles, weights)

    # For each box, test both configurations:
    # Config A: Keep original size and yaw
    # Config B: Swap width/height and rotate yaw by +/-90 degrees

    aligned_sizes = sizes.clone()
    aligned_yaws = yaw_angles.clone()

    for i in range(M):
        yaw = yaw_angles[i].item()
        size = sizes[i]  # (3,) [width, height, depth]

        # Config A: Original orientation
        diff_a = angular_distance(yaw, ref_yaw.item())

        # Config B: Swapped dimensions, rotated +/-90 degrees
        # Try both +90 and -90 and pick closer one
        yaw_plus_90 = yaw + math.pi / 2
        yaw_minus_90 = yaw - math.pi / 2

        diff_plus_90 = angular_distance(yaw_plus_90, ref_yaw.item())
        diff_minus_90 = angular_distance(yaw_minus_90, ref_yaw.item())

        # Choose best rotation adjustment
        if diff_plus_90 < diff_minus_90:
            yaw_b = yaw_plus_90
            diff_b = diff_plus_90
        else:
            yaw_b = yaw_minus_90
            diff_b = diff_minus_90

        # If swapped config is better aligned, apply the swap
        if diff_b < diff_a:
            # Swap width and height
            aligned_sizes[i] = torch.tensor([size[1], size[0], size[2]], device=device)
            aligned_yaws[i] = yaw_b

    return aligned_sizes, aligned_yaws


@dataclass
class FusedInstance:
    """A fused, static 3D instance as ObbTW."""

    # Fused ObbTW instance
    obb: ObbTW
    # Number of detections that contributed to this instance
    support_count: int
    # Indices of detections that were merged
    detection_indices: List[int]


def _load_cached_text_embeddings():
    """Try to load cached text embeddings from OWL checkpoint cache files.

    Returns:
        Dict mapping text string -> embedding tensor, or None if no cache found.
    """
    import glob as glob_mod

    ckpts_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ckpts"
    )
    cache_files = glob_mod.glob(os.path.join(ckpts_dir, "owlv2-*_textemb_*.pt"))
    if not cache_files:
        return None
    cached = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    if (
        not isinstance(cached, dict)
        or "prompts" not in cached
        or "embeddings" not in cached
    ):
        return None
    prompts = cached["prompts"]
    embeddings = cached["embeddings"]
    return {text: embeddings[i] for i, text in enumerate(prompts)}


def precompute_semantic_embeddings(
    obbs: ObbTW, device: Optional[str] = None
) -> torch.Tensor:
    """Precompute semantic embeddings for all OBBs.

    Tries to load from OWL's cached text embeddings first (avoids loading
    the 881MB checkpoint). Falls back to TextEmbedder if cache miss.

    Args:
        obbs: ObbTW tensor containing all detections
        device: Device to use for computation ("cuda", "mps", or "cpu"). If None, auto-detects.

    Returns:
        Tensor of shape (N, embed_dim) with normalized embeddings
    """
    if len(obbs) == 0:
        return torch.empty(0, 512)

    # Get all text labels and find unique ones
    text_strings = obbs.text_string()
    total_count = len(text_strings)
    unique_texts = []
    text_to_idx = {}
    text_indices = []
    for text in text_strings:
        if text not in text_to_idx:
            text_to_idx[text] = len(unique_texts)
            unique_texts.append(text)
        text_indices.append(text_to_idx[text])
    unique_count = len(unique_texts)

    print(f"Semantic embeddings: {total_count} detections, {unique_count} unique texts")

    # Try cached embeddings from OWL
    cache = _load_cached_text_embeddings()
    if cache is not None:
        missing = [t for t in unique_texts if t not in cache]
        if not missing:
            print(f"  Loaded all {unique_count} embeddings from OWL cache")
            unique_embeddings = torch.stack([cache[t] for t in unique_texts])
            text_indices_tensor = torch.tensor(text_indices, dtype=torch.long)
            return unique_embeddings[text_indices_tensor]
        else:
            print(
                f"  Cache miss for {len(missing)} texts, falling back to TextEmbedder"
            )

    # Fallback: load TextEmbedder (loads 881MB checkpoint)
    from owl.clip_tokenizer import TextEmbedder

    model = TextEmbedder()
    unique_embeddings = model.forward(unique_texts)
    text_indices_tensor = torch.tensor(text_indices, dtype=torch.long)
    embeddings = unique_embeddings[text_indices_tensor]
    print(f"  Computed {unique_count} embeddings via TextEmbedder")
    return embeddings


class BoundingBox3DFuser:
    """Fuses ObbTW detections into static instances using 3D IoU."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        min_detections: int = 4,
        samp_per_dim: int = 8,
        confidence_weighting: str = "robust",
        semantic_threshold: float = 0.7,
        enable_nms: bool = False,
        nms_iou_threshold: float = 0.6,
        conf_threshold: float = 0.55,
        shower_fixture_conf_threshold: float = 0.4,
        extent_method: str = "mean",
        envelope_quantile: float = 0.4,
        envelope_padding_m: float = 0.0,
        extent_iou_threshold: float = 0.1,
        face_min_support: int = 3,
        face_tolerance_m: float = 0.15,
    ) -> None:
        """
        Initialize 3D box fusion system.

        Args:
            iou_threshold: Minimum 3D IoU to consider boxes as potential matches
            min_detections: Minimum number of detections required to create an instance
            confidence_weighting: confidence_weighting for confidence weighting (uniform, linear, quadratic)
            semantic_threshold: Minimum semantic similarity to allow merging (hard cutoff)
            enable_nms: If True, apply NMS to fused boxes with high IoU and semantic similarity
            nms_iou_threshold: IoU threshold for NMS (boxes with IoU > this are redundant)
            conf_threshold: Minimum confidence threshold to keep detections (default: 0.55)
            shower_fixture_conf_threshold: Confidence threshold used only for
                shower-fixture detections (default: 0.4)
            extent_method: ``mean`` for the original size average, or
                ``robust_envelope`` to estimate lower/upper faces independently
            envelope_quantile: Robust-envelope trim quantile in [0, 0.5]
            envelope_padding_m: Metric padding on every robust-envelope face
            extent_iou_threshold: Relaxed IoU used to associate same-label
                detections as extent-only evidence
            face_min_support: Independent frames required to accept an outer face
            face_tolerance_m: Maximum face-position difference for consensus
        """
        self.iou_threshold = iou_threshold
        self.min_detections = min_detections
        self.confidence_weighting = confidence_weighting
        self.samp_per_dim = samp_per_dim
        self.semantic_threshold = semantic_threshold
        self.enable_nms = enable_nms
        self.nms_iou_threshold = nms_iou_threshold
        self.conf_threshold = conf_threshold
        self.shower_fixture_conf_threshold = shower_fixture_conf_threshold
        if not 0.0 <= shower_fixture_conf_threshold <= 1.0:
            raise ValueError("shower_fixture_conf_threshold must be in [0, 1]")
        if extent_method not in ("mean", "robust_envelope", "consensus_envelope"):
            raise ValueError(f"Unknown extent method: {extent_method}")
        if not 0.0 <= envelope_quantile <= 0.5:
            raise ValueError("envelope_quantile must be in [0, 0.5]")
        if envelope_padding_m < 0.0:
            raise ValueError("envelope_padding_m must be nonnegative")
        if not 0.0 <= extent_iou_threshold <= 1.0:
            raise ValueError("extent_iou_threshold must be in [0, 1]")
        if face_min_support < 1:
            raise ValueError("face_min_support must be positive")
        if face_tolerance_m < 0.0:
            raise ValueError("face_tolerance_m must be nonnegative")
        self.extent_method = extent_method
        self.envelope_quantile = envelope_quantile
        self.envelope_padding_m = envelope_padding_m
        self.extent_iou_threshold = extent_iou_threshold
        self.face_min_support = face_min_support
        self.face_tolerance_m = face_tolerance_m

    def fuse(
        self,
        detections: ObbTW,
        semantic_embeddings: Optional[torch.Tensor] = None,
        detection_group_ids: Optional[torch.Tensor] = None,
    ) -> List[FusedInstance]:
        """
        Fuse ObbTW detections into static instances.

        Args:
            detections: ObbTW tensor of shape (N, 165) containing N detections
            semantic_embeddings: Optional tensor of shape (N, D) with normalized embeddings
            detection_group_ids: Optional frame/timestamp ID for each detection

        Returns:
            List of fused instances
        """
        overall_start = time.time()

        assert isinstance(detections, ObbTW), "Detections must be ObbTW tensor"
        assert detections.ndim == 2, "Detections must be 2D tensor (N, 165)"

        n = detections.shape[0]
        if n == 0:
            return []
        if detection_group_ids is None:
            detection_group_ids = torch.arange(n)
        elif detection_group_ids.shape != (n,):
            raise ValueError("detection_group_ids must have shape (N,)")

        # Step 0: Filter by confidence threshold
        if self.conf_threshold > 0 or self.shower_fixture_conf_threshold > 0:
            labels = detections.text_string()
            thresholds = torch.tensor(
                [
                    self.shower_fixture_conf_threshold
                    if label == "shower fixture"
                    else self.conf_threshold
                    for label in labels
                ],
                dtype=detections.prob.dtype,
                device=detections.prob.device,
            )
            conf_mask = detections.prob.reshape(-1) >= thresholds
            n_before = n
            detections = detections[conf_mask]
            detection_group_ids = detection_group_ids[conf_mask]
            if semantic_embeddings is not None:
                semantic_embeddings = semantic_embeddings[conf_mask]
            n = detections.shape[0]
            print(
                f"Filtered {n_before - n} detections below confidence thresholds "
                f"(default={self.conf_threshold}, shower fixture="
                f"{self.shower_fixture_conf_threshold}) "
                f"({n_before} -> {n})"
            )
            if n == 0:
                return []

        print(f"\n{'=' * 60}")
        print(f"FUSION PERFORMANCE BREAKDOWN ({n} detections)")
        print(f"{'=' * 60}")

        # Step 1: Compute pairwise 3D IoU matrix
        print("\n[1/4] Computing 3D IoU matrix...")
        step1_start = time.time()

        if torch.backends.mps.is_available():
            detections = detections.to("mps")
        elif torch.cuda.is_available():
            detections = detections.to("cuda")

        # Auto-detect whether to use sparse IoU based on memory requirements
        estimated_memory_gb = (n * n * 4) / (1024**3)
        use_sparse = estimated_memory_gb > 4.0  # Use sparse if >4GB needed

        if use_sparse:
            print(
                f"  Using SPARSE IoU (dense would require {estimated_memory_gb:.1f}GB)"
            )
            iou_matrix = iou_mc7_sparse(
                detections,
                detections,
                samp_per_dim=self.samp_per_dim,
                iou_threshold=self.iou_threshold,  # Only store values we care about
                verbose=True,
            )
        else:
            iou_matrix = iou_mc7(
                detections,
                detections,
                samp_per_dim=self.samp_per_dim,
                verbose=True,
            )

        detections = detections.to("cpu")  # Move back to CPU
        # Move IoU matrix to CPU for combining with semantic matrix
        if iou_matrix.is_sparse:
            iou_matrix = iou_matrix.coalesce().cpu()
        else:
            iou_matrix = iou_matrix.to("cpu")

        step1_time = time.time() - step1_start
        print(f"  ✓ IoU matrix: {step1_time:.3f}s")

        # Step 1.5: Compute semantic similarity matrix if embeddings provided
        semantic_matrix = None
        step1_5_time = 0.0
        if semantic_embeddings is not None:
            print("\n[1.5/4] Computing semantic similarity matrix...")
            step1_5_start = time.time()
            # Cosine similarity: embeddings are already normalized
            semantic_matrix = torch.mm(
                semantic_embeddings, semantic_embeddings.t()
            )  # (N, N)
            step1_5_time = time.time() - step1_5_start
            print(f"  ✓ Semantic matrix: {step1_5_time:.3f}s")

        # Step 2: Cluster detections using combined similarity
        print("\n[2/4] Clustering detections...")
        step2_start = time.time()
        clusters = self._cluster_detections(iou_matrix, semantic_matrix)
        step2_time = time.time() - step2_start
        print(f"  ✓ Found {len(clusters)} clusters: {step2_time:.3f}s")

        # Step 3: Fuse boxes within each cluster
        print("\n[3/4] Fusing clusters...")
        step3_start = time.time()
        extent_clusters = None
        if self.extent_method == "consensus_envelope":
            if iou_matrix.is_sparse:
                print("  Consensus extent association unavailable for sparse IoU; using primary clusters")
                extent_clusters = clusters
            else:
                labels = detections.text_string()
                extent_clusters = []
                for cluster in clusters:
                    cluster_labels = [labels[index] for index in cluster]
                    label = max(set(cluster_labels), key=cluster_labels.count)
                    overlap = iou_matrix[:, cluster].max(dim=1).values
                    candidates = [
                        index
                        for index in range(n)
                        if labels[index] == label
                        and float(overlap[index]) >= self.extent_iou_threshold
                    ]
                    extent_clusters.append(sorted(set(cluster) | set(candidates)))
        instances = self._fuse_clusters(
            detections,
            clusters,
            extent_clusters=extent_clusters,
            detection_group_ids=detection_group_ids,
        )
        step3_time = time.time() - step3_start
        print(f"  ✓ Fused {len(instances)} instances: {step3_time:.3f}s")

        # Step 4: Filter by minimum detection count
        print("\n[4/4] Filtering by minimum detections...")
        step4_start = time.time()
        instances = [
            inst for inst in instances if inst.support_count >= self.min_detections
        ]
        step4_time = time.time() - step4_start
        print(f"  ✓ {len(instances)} instances after filtering: {step4_time:.3f}s")

        # Step 5: Optional NMS on fused boxes
        step5_time = 0.0
        if self.enable_nms and len(instances) > 0:
            print("\n[5/5] Applying NMS to fused boxes...")
            step5_start = time.time()
            instances = self._apply_nms_to_fused(instances)
            step5_time = time.time() - step5_start
            print(f"  ✓ {len(instances)} instances after NMS: {step5_time:.3f}s")

        overall_time = time.time() - overall_start

        print("\n" + "=" * 60)
        print("TIMING SUMMARY:")
        print(
            f"  1. IoU computation:     {step1_time:6.3f}s ({step1_time / overall_time * 100:5.1f}%)"
        )
        if semantic_embeddings is not None:
            print(
                f"  1.5 Semantic similarity: {step1_5_time:6.3f}s ({step1_5_time / overall_time * 100:5.1f}%)"
            )
        print(
            f"  2. Clustering:          {step2_time:6.3f}s ({step2_time / overall_time * 100:5.1f}%)"
        )
        print(
            f"  3. Fusion:              {step3_time:6.3f}s ({step3_time / overall_time * 100:5.1f}%)"
        )
        print(
            f"  4. Filtering:           {step4_time:6.3f}s ({step4_time / overall_time * 100:5.1f}%)"
        )
        if self.enable_nms:
            print(
                f"  5. NMS:                 {step5_time:6.3f}s ({step5_time / overall_time * 100:5.1f}%)"
            )
        print("  ----------------------------------")
        print(f"  TOTAL:                  {overall_time:6.3f}s")
        print("=" * 60 + "\n")

        return instances

    def _cluster_detections(
        self, iou_matrix: torch.Tensor, semantic_matrix: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Cluster detections using connected components on combined similarity graph.

        Handles both dense and sparse IoU matrices.

        Args:
            iou_matrix: NxN IoU matrix (dense or sparse PyTorch tensor)
            semantic_matrix: Optional NxN semantic similarity matrix (cosine similarity)

        Returns:
            List of clusters, where each cluster is a list of detection indices
        """
        is_sparse = iou_matrix.is_sparse

        # Collect edges (pairs of indices that should be in the same cluster)
        edges: list[tuple[int, int]] = []

        if is_sparse:
            # Sparse path: IoU matrix is already thresholded from iou_mc7_sparse
            iou_coo = iou_matrix.coalesce()
            indices = iou_coo.indices()  # (2, nnz)
            values = iou_coo.values()  # (nnz,)
            n = iou_matrix.shape[0]

            if semantic_matrix is not None:
                # Look up semantic similarities only for non-zero IoU pairs
                rows, cols = indices[0], indices[1]
                semantic_values = semantic_matrix[rows, cols]

                # Filter by semantic threshold
                keep_mask = semantic_values >= self.semantic_threshold

                # Count blocked pairs for logging
                total_pairs = len(values)
                blocked_pairs = (~keep_mask).sum().item()
                print(
                    f"  Semantic cutoff at {self.semantic_threshold:.2f}: blocked {blocked_pairs}/{total_pairs} potential merges"
                )

                # Filter indices
                indices = indices[:, keep_mask]

            for i in range(indices.shape[1]):
                edges.append((indices[0, i].item(), indices[1, i].item()))

        else:
            # Dense path (original code)
            n = iou_matrix.shape[0]

            # Apply hard semantic cutoff if semantic matrix provided
            if semantic_matrix is not None:
                semantic_mask = (
                    semantic_matrix.cpu() >= self.semantic_threshold
                )  # (N, N) bool
                combined_matrix = iou_matrix * semantic_mask.float()

                # Count how many pairs were blocked
                total_pairs = (iou_matrix >= self.iou_threshold).sum().item()
                blocked_pairs = (
                    ((iou_matrix >= self.iou_threshold) & ~semantic_mask.cpu())
                    .sum()
                    .item()
                )
                print(
                    f"  Semantic cutoff at {self.semantic_threshold:.2f}: blocked {blocked_pairs}/{total_pairs} potential merges"
                )
            else:
                combined_matrix = iou_matrix

            # Find edges from adjacency
            adj_np = combined_matrix.cpu().numpy() >= self.iou_threshold
            rows, cols = adj_np.nonzero()
            for r, c in zip(rows, cols):
                edges.append((int(r), int(c)))

        # Union-Find to compute connected components
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in edges:
            union(a, b)

        # Group by root
        clusters_dict: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters_dict.setdefault(root, []).append(i)
        clusters: list[list[int]] = list(clusters_dict.values())

        # Filter out empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        print(f"Found {len(clusters)} clusters from similarity graph")

        return clusters

    def _fuse_clusters(
        self,
        detections: ObbTW,
        clusters: list[list[int]],
        extent_clusters: Optional[list[list[int]]] = None,
        detection_group_ids: Optional[torch.Tensor] = None,
    ) -> list[FusedInstance]:
        """
        Fuse detections within each cluster into single instances.

        Uses confidence-weighted averaging for pose (7 DoF) and size.
        Accounts for 90-degree rotation symmetry by aligning box dimensions.

        Args:
            detections: All detections (N, 165)
            clusters: List of clusters (each is list of detection indices)

        Returns:
            List of fused instances
        """
        instances = []

        if extent_clusters is None:
            extent_clusters = clusters
        if detection_group_ids is None:
            detection_group_ids = torch.arange(len(detections))

        for cluster_index, cluster in enumerate(clusters):
            cluster_detections = detections[cluster]  # (M, 165)

            # Extract sizes and yaw angles
            extents = cluster_detections.bb3_object  # (M, 6)
            sizex = extents[:, 1] - extents[:, 0]  # (M,)
            sizey = extents[:, 3] - extents[:, 2]  # (M,)
            sizez = extents[:, 5] - extents[:, 4]  # (M,)
            sizes = torch.stack([sizex, sizey, sizez], dim=1)  # (M, 3)

            # Get yaw angles from poses
            poses = cluster_detections.T_world_object  # List of M PoseTW
            eulers = poses.to_euler()  # (M, 3)
            yaw_angles = eulers[:, 2]  # (M,)

            # STEP 1: Align boxes to canonical orientation (accounts for 90° rotations)
            # Use base confidence weights for alignment reference
            base_confidences = cluster_detections.prob.reshape(-1)  # (M,)
            base_weights = base_confidences / (base_confidences.sum() + 1e-8)

            aligned_sizes, aligned_yaws = self._align_boxes_r90(
                sizes, yaw_angles, base_weights
            )

            # STEP 2: Compute weights based on ALIGNED sizes/yaws (for robust mode)
            # This ensures outlier detection happens after 90° alignment
            weights = self._compute_fusion_weights_with_alignment(
                cluster_detections, aligned_sizes, aligned_yaws
            )

            # STEP 3: Fuse aligned yaw angles (weighted average with 180° symmetry)
            mean_yaw, _ = self._weighted_yaw_mean(aligned_yaws, weights)

            # Fuse translations and create the reference pose.
            translations = torch.stack([pose.t for pose in poses])  # (M, 3)
            weights_t = weights.view(-1, 1).expand_as(translations)
            fused_translation = (translations * weights_t).sum(dim=0)  # (3,)
            new_eulers = torch.tensor([0, 0, mean_yaw]).to(fused_translation)  # (3,)
            new_eulers = new_eulers.reshape(1, 3)  # (1, 3)
            fused_rotation = rotation_from_euler(new_eulers)[0]

            # STEP 4: Estimate box extents. The robust-envelope mode projects
            # every contributing box into the fused orientation and estimates
            # its lower and upper faces independently. This avoids shrinking a
            # long object when partial detections have shifted centers.
            if self.extent_method in ("robust_envelope", "consensus_envelope"):
                extent_indices = extent_clusters[cluster_index]
                extent_detections = detections[extent_indices]
                corners_world = extent_detections.bb3corners_world
                corners_local = torch.matmul(
                    corners_world - fused_translation.view(1, 1, 3),
                    fused_rotation,
                )
                lower_faces = corners_local.min(dim=1).values
                upper_faces = corners_local.max(dim=1).values
                if self.extent_method == "consensus_envelope":
                    primary_corners_world = cluster_detections.bb3corners_world
                    primary_corners_local = torch.matmul(
                        primary_corners_world - fused_translation.view(1, 1, 3),
                        fused_rotation,
                    )
                    primary_lower = primary_corners_local.min(dim=1).values
                    primary_upper = primary_corners_local.max(dim=1).values
                    fallback_lower = weighted_quantile(
                        primary_lower, weights, self.envelope_quantile
                    )
                    fallback_upper = weighted_quantile(
                        primary_upper, weights, 1.0 - self.envelope_quantile
                    )
                    extent_group_ids = detection_group_ids[extent_indices]
                    # Consensus expansion is intentionally limited to the
                    # dominant object axis. Applying outer-face evidence to
                    # depth and height can absorb walls, counters, or floors
                    # when the 3D center is noisy across viewpoints.
                    major_axis = int(
                        torch.argmax(fallback_upper - fallback_lower).item()
                    )
                    lower = fallback_lower.clone()
                    upper = fallback_upper.clone()
                    consensus_lower = consensus_outer_face(
                        lower_faces[:, major_axis],
                        extent_group_ids,
                        choose_lower=True,
                        min_support=self.face_min_support,
                        tolerance_m=self.face_tolerance_m,
                        fallback=fallback_lower[major_axis],
                    )
                    consensus_upper = consensus_outer_face(
                        upper_faces[:, major_axis],
                        extent_group_ids,
                        choose_lower=False,
                        min_support=self.face_min_support,
                        tolerance_m=self.face_tolerance_m,
                        fallback=fallback_upper[major_axis],
                    )
                    # Consensus may expand the robust envelope, never shrink it.
                    lower[major_axis] = torch.minimum(
                        consensus_lower, fallback_lower[major_axis]
                    )
                    upper[major_axis] = torch.maximum(
                        consensus_upper, fallback_upper[major_axis]
                    )
                else:
                    lower = weighted_quantile(
                        lower_faces, weights, self.envelope_quantile
                    )
                    upper = weighted_quantile(
                        upper_faces, weights, 1.0 - self.envelope_quantile
                    )
                lower = lower - self.envelope_padding_m
                upper = upper + self.envelope_padding_m
                fused_sizes = (upper - lower).clamp_min(1e-4)
                local_center = 0.5 * (lower + upper)
                fused_translation = fused_translation + torch.matmul(
                    fused_rotation, local_center
                )
            else:
                weights_sizes = weights.view(-1, 1).expand_as(aligned_sizes)
                fused_sizes = (aligned_sizes * weights_sizes).sum(dim=0)

            bb3_object = torch.stack(
                [
                    -fused_sizes[0] / 2,
                    fused_sizes[0] / 2,
                    -fused_sizes[1] / 2,
                    fused_sizes[1] / 2,
                    -fused_sizes[2] / 2,
                    fused_sizes[2] / 2,
                ]
            )
            fused_pose = PoseTW.from_Rt(fused_rotation, fused_translation)

            # Fuse confidence (weighted average)
            probs = cluster_detections.prob  # (M, 1)
            weights_prob = weights.view(-1, 1)
            fused_prob = (probs * weights_prob).sum(dim=0)  # (1,)

            # Take most common text label.
            all_text_labels = cluster_detections.text_string()
            text_label = max(set(all_text_labels), key=all_text_labels.count)
            text_padded = string2tensor(pad_string(text_label, max_len=128))

            # Take most common semantic label.
            all_semantic_labels = cluster_detections.sem_id
            sem_id = all_semantic_labels[all_semantic_labels.argsort()[-1]]

            # Create fused ObbTW from pose and size
            fused_obb = ObbTW.from_lmc(
                bb3_object=bb3_object,
                prob=fused_prob.unsqueeze(0),  # (1, 1)
                T_world_object=fused_pose,
                text=text_padded,
                sem_id=sem_id,
            )

            instances.append(
                FusedInstance(
                    obb=fused_obb,
                    support_count=len(cluster),
                    detection_indices=cluster,
                )
            )

        return instances

    def _weighted_yaw_mean(
        self, angles: torch.Tensor, weights: torch.Tensor, eps: float = 1e-8
    ):
        """Weighted mean of 1D rotations with 180-degree symmetry (pi-periodic)."""
        return weighted_yaw_mean(angles, weights, eps)

    def _align_boxes_r90(
        self,
        sizes: torch.Tensor,
        yaw_angles: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align boxes to canonical orientation accounting for 90-degree rotation symmetry."""
        return align_boxes_r90(sizes, yaw_angles, weights)

    def _angular_distance(self, angle1: float, angle2: float) -> float:
        """Compute smallest angular distance between two angles accounting for 180-degree symmetry."""
        return angular_distance(angle1, angle2)

    def _compute_fusion_weights_with_alignment(
        self,
        detections: ObbTW,
        aligned_sizes: torch.Tensor,
        aligned_yaws: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weights based on ALIGNED sizes and yaws for robust mode.

        This ensures outlier detection happens AFTER 90° alignment, preventing
        valid boxes with swapped dimensions from being incorrectly marked as outliers.

        Args:
            detections: Original detections (for positions and confidence)
            aligned_sizes: (M, 3) tensor of aligned sizes
            aligned_yaws: (M,) tensor of aligned yaw angles

        Returns:
            Torch tensor of weights (sums to 1)
        """
        confidences = detections.prob.reshape(-1)  # (M,)

        if self.confidence_weighting == "uniform":
            weights = torch.ones_like(confidences)
        elif self.confidence_weighting == "linear":
            weights = confidences
        elif self.confidence_weighting == "quadratic":
            weights = confidences**2
        elif self.confidence_weighting == "robust":
            # RANSAC-like robust weighting using ALIGNED data
            weights = self._compute_robust_weights_aligned(
                detections, aligned_sizes, aligned_yaws, confidences
            )
        else:
            raise ValueError(
                f"Unknown confidence weighting: {self.confidence_weighting}"
            )

        # Normalize to sum to 1
        weights = weights / (weights.sum() + 1e-8)

        return weights

    def _compute_robust_weights_aligned(
        self,
        detections: ObbTW,
        aligned_sizes: torch.Tensor,
        aligned_yaws: torch.Tensor,
        confidences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute robust weights using ALIGNED sizes and yaws.

        This fixes the bug where boxes with 90° rotation differences were
        incorrectly marked as outliers before alignment.

        Args:
            detections: ObbTW detections in cluster (M, 165)
            aligned_sizes: (M, 3) tensor of aligned sizes
            aligned_yaws: (M,) tensor of aligned yaw angles
            confidences: Base confidence scores (M,)

        Returns:
            Robust weights (M,) that down-weight outliers
        """
        M = len(detections)
        if M <= 2:
            # Not enough data for robust statistics, use confidence only
            return confidences

        # Extract positions (alignment doesn't affect positions)
        poses = detections.T_world_object
        translations = torch.stack([pose.t for pose in poses])  # (M, 3)

        # Use ALIGNED sizes and yaws for outlier detection
        sizes = aligned_sizes  # (M, 3)
        yaw_angles = aligned_yaws  # (M,)

        # Compute robust statistics using median
        median_position = torch.median(translations, dim=0).values  # (3,)
        median_size = torch.median(sizes, dim=0).values  # (3,)

        # For yaw: use circular mean with 180° symmetry
        mean_yaw, _ = self._weighted_yaw_mean(yaw_angles, torch.ones_like(yaw_angles))

        # Compute deviations
        position_deviations = torch.norm(translations - median_position, dim=1)  # (M,)
        size_deviations = torch.norm(sizes - median_size, dim=1)  # (M,)

        # Yaw deviations accounting for 180° symmetry
        yaw_deviations = torch.tensor(
            [self._angular_distance(yaw.item(), mean_yaw.item()) for yaw in yaw_angles],
            device=yaw_angles.device,
        )

        # Compute MAD (Median Absolute Deviation) for robust outlier detection
        mad_position = torch.median(position_deviations)
        mad_size = torch.median(size_deviations)
        mad_yaw = torch.median(yaw_deviations)

        # Avoid division by zero
        mad_position = max(mad_position, 1e-6)
        mad_size = max(mad_size, 1e-6)
        mad_yaw = max(mad_yaw, 1e-6)

        # Normalized deviations (similar to z-scores but robust)
        # Scale factor 1.4826 makes MAD consistent with std for normal distribution
        scale = 1.4826
        normalized_position = position_deviations / (scale * mad_position)
        normalized_size = size_deviations / (scale * mad_size)
        normalized_yaw = yaw_deviations / (scale * mad_yaw)

        # Combined outlier score (higher = more likely to be outlier)
        outlier_score = (normalized_position + normalized_size + normalized_yaw) / 3.0

        # Convert outlier score to weights using Huber-like function
        # Detections with outlier_score > threshold are down-weighted
        threshold = 2.5  # Similar to 2.5 sigma in normal distribution
        inlier_weights = torch.where(
            outlier_score <= threshold,
            torch.ones_like(outlier_score),  # Inliers get full weight
            threshold / outlier_score,  # Outliers get reduced weight
        )

        # Combine with original confidence scores
        robust_weights = confidences * inlier_weights

        return robust_weights

    def _apply_nms_to_fused(
        self,
        instances: list[FusedInstance],
    ) -> list[FusedInstance]:
        """Apply NMS to fused boxes to remove redundant instances based on IoU only.

        Args:
            instances: List of fused instances

        Returns:
            Filtered list of instances after NMS
        """
        return apply_nms_to_fused_instances(instances, self.nms_iou_threshold)


def apply_nms_to_fused_instances(
    instances: list[FusedInstance],
    nms_iou_threshold: float = 0.6,
) -> list[FusedInstance]:
    """Apply NMS to fused instances to remove redundant boxes based on IoU.

    This is a standalone function that can be used to apply NMS to a list of
    fused instances outside of the BoundingBox3DFuser class.

    Args:
        instances: List of fused instances
        nms_iou_threshold: IoU threshold for NMS (boxes with IoU > this are redundant)

    Returns:
        Filtered list of instances after NMS
    """
    if len(instances) <= 1:
        return instances

    # Compute pairwise IoU for all fused boxes
    fused_obbs = torch.stack([inst.obb for inst in instances])
    if torch.cuda.is_available():
        fused_obbs = fused_obbs.to("cuda")
    elif torch.backends.mps.is_available():
        fused_obbs = fused_obbs.to("mps")

    # Sample more accurately here.
    iou_matrix = iou_mc7(fused_obbs, fused_obbs, samp_per_dim=32, verbose=False).cpu()

    # Greedily remove instances with high IoU overlap
    keep_mask = torch.ones(len(instances), dtype=torch.bool)
    for i in range(len(instances)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(instances)):
            if not keep_mask[j]:
                continue
            if iou_matrix[i, j] > nms_iou_threshold:
                # Get class names for debug output
                class_i = instances[i].obb.text_string()
                class_j = instances[j].obb.text_string()

                # Remove instance with lower support count
                if instances[i].support_count >= instances[j].support_count:
                    keep_mask[j] = False
                    print(
                        f"  Removing instance {j} '{class_j}' (support={instances[j].support_count}, IoU={iou_matrix[i, j].item():.3f}) "
                        f"in favor of {i} '{class_i}' (support={instances[i].support_count})"
                    )
                else:
                    keep_mask[i] = False
                    print(
                        f"  Removing instance {i} '{class_i}' (support={instances[i].support_count}, IoU={iou_matrix[i, j].item():.3f}) "
                        f"in favor of {j} '{class_j}' (support={instances[j].support_count})"
                    )
                    break

    filtered = [inst for idx, inst in enumerate(instances) if keep_mask[idx]]
    print(f"  NMS: Removed {len(instances) - len(filtered)} redundant instances")
    return filtered


def fuse_obbs_from_csv(
    input_path: str,
    output_path: Optional[str] = None,
    iou_threshold: float = 0.3,
    min_detections: int = 4,
    conf_threshold: float = 0.55,
    shower_fixture_conf_threshold: float = 0.4,
    extent_method: str = "mean",
    envelope_quantile: float = 0.4,
    envelope_padding_m: float = 0.0,
    extent_iou_threshold: float = 0.1,
    face_min_support: int = 3,
    face_tolerance_m: float = 0.15,
    point_cloud_path: str | None = None,
) -> list[FusedInstance]:
    """
    Load OBBs from a CSV file, fuse them, and save the results.

    Args:
        input_path: Path to input obb.csv file
        output_path: Path to output obb_fused.csv file (default: input with _fused suffix)
        iou_threshold: IoU threshold for 3D box fusion
        min_detections: Minimum number of detections required to create an instance
        conf_threshold: Minimum confidence threshold to filter detections
        shower_fixture_conf_threshold: Confidence threshold used only for
            shower-fixture detections
        extent_method: Fused extent estimator (``mean`` or ``robust_envelope``)
        envelope_quantile: Trim quantile for robust lower/upper box faces
        envelope_padding_m: Padding added to each robust-envelope face
        extent_iou_threshold: Relaxed same-label overlap for extent evidence
        face_min_support: Independent timestamps required per outer face
        face_tolerance_m: Face consensus tolerance in metres
        point_cloud_path: Aligned PLY used to derive axis-aligned SpatialLM
            boxes from the fused object point subsets

    Returns:
        List of fused instances
    """
    # Determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_fused{ext}"
    spatiallm_output_path = os.path.join(
        os.path.dirname(output_path) or ".", "spatiallm_bboxes.txt"
    )

    print(f"==> Loading OBBs from {input_path}")
    timed_obbs = read_obb_csv(input_path)

    if len(timed_obbs) == 0:
        print("==> No OBBs found in input file, nothing to fuse")
        write_spatiallm_bboxes(ObbTW(torch.zeros(0, 165)), spatiallm_output_path)
        return []

    # Concatenate all OBBs from all timestamps
    all_obbs_list = list(timed_obbs.values())
    all_obbs = torch.cat(all_obbs_list, dim=0)
    detection_group_ids = torch.cat(
        [torch.full((len(obbs),), index, dtype=torch.long) for index, obbs in enumerate(all_obbs_list)]
    )
    print(f"==> Loaded {all_obbs.shape[0]} OBBs from {len(timed_obbs)} timestamps")

    # Create fuser and run fusion
    print(f"\n{'=' * 60}")
    print(f"FUSING {all_obbs.shape[0]} OBBs")
    print(f"{'=' * 60}")

    fuser = BoundingBox3DFuser(
        iou_threshold=iou_threshold,
        min_detections=min_detections,
        conf_threshold=conf_threshold,
        shower_fixture_conf_threshold=shower_fixture_conf_threshold,
        extent_method=extent_method,
        envelope_quantile=envelope_quantile,
        envelope_padding_m=envelope_padding_m,
        extent_iou_threshold=extent_iou_threshold,
        face_min_support=face_min_support,
        face_tolerance_m=face_tolerance_m,
    )
    fused_instances = fuser.fuse(all_obbs, detection_group_ids=detection_group_ids)
    print(f"==> Fused into {len(fused_instances)} static instances")

    if len(fused_instances) == 0:
        write_spatiallm_bboxes(ObbTW(torch.zeros(0, 165)), spatiallm_output_path)
        print(
            "==> No fused instances produced; saved an empty SpatialLM file to "
            f"{spatiallm_output_path}"
        )
        return []

    # Extract OBBs from fused instances
    fused_obbs = torch.stack([inst.obb for inst in fused_instances], dim=0)

    # Round prob to 2 decimal places to avoid floating point artifacts in CSV
    rounded_prob = torch.round(fused_obbs.prob * 100) / 100
    fused_obbs.set_prob(rounded_prob.squeeze(-1), use_mask=False)

    # Build sem_id_to_name mapping from the fused OBBs
    sem_id_to_name = {}
    for obb in fused_obbs:
        sem_id = int(obb.sem_id.item())
        if sem_id not in sem_id_to_name:
            text = unpad_string(tensor2string(obb.text.int()))
        else:
            text = sem_id_to_name[sem_id]
        sem_id_to_name[sem_id] = text

    # Write fused OBBs with timestamp 0 (static map)
    writer = ObbCsvWriter2(output_path)
    writer.write(fused_obbs, timestamps_ns=0, sem_id_to_name=sem_id_to_name)
    writer.close()
    print(f"==> Saved {len(fused_instances)} fused OBBs to {output_path}")

    spatiallm_lines = write_spatiallm_bboxes(
        fused_obbs,
        spatiallm_output_path,
        point_cloud_path=point_cloud_path,
    )
    print(f"==> Saved SpatialLM bounding boxes to {spatiallm_output_path}")
    for line in spatiallm_lines:
        print(line)

    return fused_instances


def main() -> None:
    """Fuse OBBs from a CSV file into static instances."""
    parser = argparse.ArgumentParser(
        description="Fuse 3D bounding box detections from CSV into static instances"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input obb.csv file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output obb_fused.csv file (default: input path with _fused suffix)",
    )
    parser.add_argument(
        "--point-cloud",
        "--point_cloud",
        dest="point_cloud",
        default=None,
        help="Aligned PLY used to write point-subset AABBs to spatiallm_bboxes.txt",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.3,
        help="IoU threshold for 3D box fusion (default: 0.3)",
    )
    parser.add_argument(
        "--min_detections",
        type=int,
        default=4,
        help="Minimum number of detections required to create an instance (default: 4)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.55,
        help="Minimum confidence threshold to filter detections (default: 0.55)",
    )
    parser.add_argument(
        "--shower-fixture-conf-threshold",
        type=float,
        default=0.4,
        help="Confidence threshold used only for shower fixtures (default: 0.4)",
    )
    parser.add_argument(
        "--extent-method",
        choices=("mean", "robust_envelope", "consensus_envelope"),
        default="mean",
        help="How to calculate fused box extents (default: mean)",
    )
    parser.add_argument(
        "--envelope-quantile",
        type=float,
        default=0.4,
        help="Robust-envelope trim quantile in [0, 0.5] (default: 0.4)",
    )
    parser.add_argument(
        "--envelope-padding-m",
        type=float,
        default=0.0,
        help="Padding added to each robust-envelope face in metres (default: 0.0)",
    )
    parser.add_argument(
        "--extent-iou-threshold",
        type=float,
        default=0.1,
        help="Relaxed same-label IoU for extent-only evidence (default: 0.1)",
    )
    parser.add_argument(
        "--face-min-support",
        type=int,
        default=3,
        help="Independent timestamps required to accept an outer face (default: 3)",
    )
    parser.add_argument(
        "--face-tolerance-m",
        type=float,
        default=0.15,
        help="Face consensus tolerance in metres (default: 0.15)",
    )
    args = parser.parse_args()

    fuse_obbs_from_csv(
        input_path=args.input,
        output_path=args.output,
        iou_threshold=args.iou,
        min_detections=args.min_detections,
        conf_threshold=args.conf_threshold,
        shower_fixture_conf_threshold=args.shower_fixture_conf_threshold,
        extent_method=args.extent_method,
        envelope_quantile=args.envelope_quantile,
        envelope_padding_m=args.envelope_padding_m,
        extent_iou_threshold=args.extent_iou_threshold,
        face_min_support=args.face_min_support,
        face_tolerance_m=args.face_tolerance_m,
        point_cloud_path=args.point_cloud,
    )


def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find minimum cost assignment using the Hungarian algorithm.

    Args:
        cost_matrix: (n, m) cost matrix. Can be rectangular.

    Returns:
        Tuple of (row_indices, col_indices) for the optimal assignment.
    """
    cost = np.array(cost_matrix, dtype=np.float64)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Transpose if more rows than columns (algorithm needs n <= m)
    transposed = False
    if n > m:
        cost = cost.T
        n, m = m, n
        transposed = True

    # Pad to square if needed
    if n < m:
        cost = np.vstack([cost, np.full((m - n, m), 0.0)])

    size = m
    u = np.zeros(size + 1)
    v = np.zeros(size + 1)
    p = np.zeros(size + 1, dtype=int)
    way = np.zeros(size + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = -1

            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Extract assignment
    row_ind = []
    col_ind = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] <= n:
            row_ind.append(p[j] - 1)
            col_ind.append(j - 1)

    row_ind = np.array(row_ind, dtype=int)
    col_ind = np.array(col_ind, dtype=int)

    if transposed:
        row_ind, col_ind = col_ind, row_ind

    # Sort by row index
    order = np.argsort(row_ind)
    return row_ind[order], col_ind[order]


if __name__ == "__main__":
    main()
