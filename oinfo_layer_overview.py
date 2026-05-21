#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR-rank heatmap visualization for HyperPotter.

Author: Qing Wen

This script visualizes high-order interactions in HyperPotter by computing
O-information and Gradient O-information over prototype-aligned hyperedge slots.

Input:
    - A trained HyperPotter checkpoint.
    - The corresponding prototype banks.
    - One or more protocol files.

Output:
    - summary.csv
    - <dataset_name>/summary.json
    - <dataset_name>/layer_details.json
    - <dataset_name>/sr_rank_heatmap_2x2.{pdf,png}

Example:
    CUDA_VISIBLE_DEVICES=0 python oinfo_layer_overview.py \
        --model-path /path/to/HyperPotter.pth \
        --prototype-banks-path /path/to/proto_banks.pt \
        --protocol-files ./protocols/InTheWild.txt \
        --dataset-names ITW \
        --splits eval \
        --output-dir ./outputs/oinfo_visualization
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm


# Keep the same visual style as the original analysis script.
sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
def import_model():
    """Import the HyperPotter/VARIA model from common project module names."""
    for module_name in ["top_varia_model", "varia_model", "varia_model_ae0", "varia_model_ae"]:
        try:
            module = __import__(module_name, fromlist=["Model"])
            return module.Model
        except Exception:
            continue
    raise ImportError(
        "Cannot import `Model`. Please make sure the project root is in PYTHONPATH "
        "and one of [top_varia_model, varia_model, varia_model_ae0, varia_model_ae] exists."
    )


def import_dataset():
    """Import the default protocol-based dataset class."""
    try:
        from data_utils import Default_dataset
        return Default_dataset
    except Exception as exc:
        raise ImportError(
            "Cannot import `Default_dataset` from `data_utils`. "
            "Please make sure the project root is in PYTHONPATH."
        ) from exc


Model = import_model()
DefaultDataset = import_dataset()


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_state_dict_flexible(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")
    print(f"[info] Model weights loaded: {checkpoint_path}")


def load_prototype_banks(model, prototype_banks_path, device):
    if not hasattr(model, "proto_manager"):
        raise RuntimeError("The model has no `proto_manager`; prototype banks are required.")

    if hasattr(model.proto_manager, "load_banks"):
        model.proto_manager.load_banks(prototype_banks_path)
        print(f"[info] Prototype banks loaded: {prototype_banks_path}")
        return

    if hasattr(model.proto_manager, "load_state_dict"):
        state_dict = torch.load(prototype_banks_path, map_location=device)
        model.proto_manager.load_state_dict(state_dict)
        print(f"[info] Prototype banks state loaded: {prototype_banks_path}")
        return

    raise RuntimeError("`proto_manager` has neither `load_banks` nor `load_state_dict`.")


def build_model(device):
    dummy_args = SimpleNamespace()
    try:
        return Model(dummy_args, device=device)
    except TypeError:
        pass
    try:
        return Model(device=device)
    except TypeError:
        pass
    return Model(dummy_args)


def force_clear_external_inits(model, layer_names):
    """Clear external FCM centroid initialization if the model keeps such states."""
    for layer_name in layer_names:
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        if hasattr(layer, "fcm_module") and hasattr(layer.fcm_module, "external_init_centroids"):
            layer.fcm_module.external_init_centroids = None


# ---------------------------------------------------------------------------
# HOI / O-information utilities
# ---------------------------------------------------------------------------
def import_hoi_metrics():
    try:
        from hoi.metrics import GradientOinfo, Oinfo
        return Oinfo, GradientOinfo
    except Exception as exc:
        raise ImportError("Failed to import `hoi`. Please install it with `pip install -U hoi`.") from exc


def as_triplets(num_variables):
    return np.array(list(combinations(range(num_variables), 3)), dtype=np.int64)


def prepare_for_hoi(X, seed=0, jitter=1e-6):
    """Z-score variables and add tiny noise for numerical stability."""
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    X = (X - mean) / (std + 1e-12)

    if jitter > 0:
        rng = np.random.default_rng(seed)
        X = X + rng.normal(scale=jitter, size=X.shape)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values remain after HOI preprocessing.")
    return X


def extract_values_and_triplets(metric_object, fit_output, num_variables, order=3):
    """Handle small output differences across hoi versions."""
    values = None

    if isinstance(fit_output, dict):
        for key in ["oinfo", "Oinfo", "o_info", "O_information", "values"]:
            if key in fit_output:
                values = np.asarray(fit_output[key]).reshape(-1)
                break

        if values is None:
            for candidate in fit_output.values():
                try:
                    arr = np.asarray(candidate)
                    if arr.dtype.kind in "fc" and arr.size > 0:
                        values = arr.reshape(-1)
                        break
                except Exception:
                    continue
    else:
        values = np.asarray(fit_output).reshape(-1)

    triplets = None
    for attr in ["multiplets", "multiplets_", "mults", "mults_"]:
        if not hasattr(metric_object, attr):
            continue
        try:
            maybe_triplets = np.asarray(getattr(metric_object, attr))
            if maybe_triplets.ndim == 2 and maybe_triplets.shape[1] == order:
                triplets = maybe_triplets
                break
        except Exception:
            pass

    if triplets is None:
        triplets = as_triplets(num_variables)
        if values is not None and len(values) != len(triplets):
            n = min(len(values), len(triplets))
            values = values[:n]
            triplets = triplets[:n]

    return values, triplets


def compute_oinfo_triplets(X, method="gc"):
    X = prepare_for_hoi(X, seed=0, jitter=1e-6)
    Oinfo, _ = import_hoi_metrics()
    metric = Oinfo(X)
    output = metric.fit(method=method, minsize=3, maxsize=3)
    return extract_values_and_triplets(metric, output, num_variables=X.shape[1], order=3)


def compute_gradient_oinfo_triplets(X, y, method="gc"):
    X = prepare_for_hoi(X, seed=1, jitter=1e-6)
    y = np.asarray(y, dtype=np.float64)
    _, GradientOinfo = import_hoi_metrics()
    metric = GradientOinfo(X, y)
    output = metric.fit(method=method, minsize=3, maxsize=3)
    return extract_values_and_triplets(metric, output, num_variables=X.shape[1], order=3)


def rankdata_average_ties(x):
    """Rank values using average ranks for ties. Ranks are in [1, n]."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)

    sorted_x = x[order]
    i = 0
    while i < len(sorted_x):
        j = i
        while j + 1 < len(sorted_x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg_rank = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg_rank
        i = j + 1

    return ranks


def minmax_normalize(x):
    x = np.asarray(x, dtype=np.float64)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def sr_rank_from_triplets(values, triplets, num_variables):
    """Aggregate triplet-wise O-information into slot-wise synergy/redundancy scores."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    triplets = np.asarray(triplets, dtype=np.int64)

    synergy_part = np.maximum(-values, 0.0)
    redundancy_part = np.maximum(values, 0.0)

    synergy_sum = np.zeros((num_variables,), dtype=np.float64)
    redundancy_sum = np.zeros((num_variables,), dtype=np.float64)
    counts = np.zeros((num_variables,), dtype=np.int64)

    for idx in range(triplets.shape[0]):
        i, j, k = triplets[idx]
        s = synergy_part[idx]
        r = redundancy_part[idx]

        synergy_sum[i] += s
        synergy_sum[j] += s
        synergy_sum[k] += s

        redundancy_sum[i] += r
        redundancy_sum[j] += r
        redundancy_sum[k] += r

        counts[i] += 1
        counts[j] += 1
        counts[k] += 1

    counts = np.maximum(counts, 1)
    synergy_score = synergy_sum / counts
    redundancy_score = redundancy_sum / counts

    rank_synergy = rankdata_average_ties(synergy_score)
    rank_redundancy = rankdata_average_ties(redundancy_score)
    sr_rank = rank_synergy - rank_redundancy

    layer_sr = float(sr_rank.mean())
    synergy_ratio = float(np.mean(values < 0))

    return layer_sr, synergy_ratio, synergy_score, redundancy_score, sr_rank


# ---------------------------------------------------------------------------
# HyperPotter feature extraction
# ---------------------------------------------------------------------------
def centroid_global_similarity(centroids, global_prototypes):
    """Return cosine similarity between each centroid and its global prototype."""
    device = centroids.device
    dtype = centroids.dtype

    c = F.normalize(centroids, dim=-1)                                      # (B, K, C)
    g = F.normalize(global_prototypes.to(device=device, dtype=dtype), dim=-1)
    g = g.unsqueeze(0)                                                       # (1, K, C)

    similarity = (c * g).sum(dim=-1)                                         # (B, K)
    return similarity.detach().cpu().numpy().astype(np.float64, copy=False)


def discover_hypergraph_layers(model, preferred_layers=None):
    default_candidates = [
        "HGNN_layer_S",
        "HGNN_layer_T",
        "HtrgHGNN_layer_ST11",
        "HtrgHGNN_layer_ST12",
        "HtrgHGNN_layer_ST21",
        "HtrgHGNN_layer_ST22",
    ]
    candidates = preferred_layers if preferred_layers else default_candidates

    found = []
    for layer_name in candidates:
        if hasattr(model, layer_name) and hasattr(getattr(model, layer_name), "fcm_module"):
            found.append(layer_name)

    return found


def register_similarity_hooks(model, layer_names, bonafide_label=1, debug=False):
    """Collect centroid-global-prototype similarities from target hypergraph layers."""
    buffers = {name: {"bonafide": [], "spoof": []} for name in layer_names}
    state = {"current_labels": None}
    warned = set()

    def make_hook(layer_name):
        def hook(_module, _inputs, outputs):
            labels = state["current_labels"]
            if labels is None:
                return

            try:
                centroids = outputs[1]  # Expected shape: (B, K, C)
            except Exception:
                return

            if not hasattr(model, "proto_manager") or not hasattr(model.proto_manager, "banks"):
                if debug and "no_proto_manager" not in warned:
                    print("[debug] no proto_manager/banks; no features will be collected.")
                    warned.add("no_proto_manager")
                return

            if layer_name not in model.proto_manager.banks:
                if debug and layer_name not in warned:
                    print(f"[debug] prototype bank missing for layer: {layer_name}")
                    warned.add(layer_name)
                return

            bank = model.proto_manager.banks[layer_name]
            global_prototypes = bank.get("global", None)
            if not isinstance(global_prototypes, torch.Tensor) or global_prototypes.numel() == 0:
                key = f"{layer_name}_empty_global_proto"
                if debug and key not in warned:
                    print(f"[debug] global prototype missing or empty for layer: {layer_name}")
                    warned.add(key)
                return

            features = centroid_global_similarity(centroids, global_prototypes)
            labels_cpu = labels.detach().cpu().view(-1).tolist()

            for i in range(features.shape[0]):
                class_name = "bonafide" if int(labels_cpu[i]) == int(bonafide_label) else "spoof"
                buffers[layer_name][class_name].append(features[i])

        return hook

    handles = []
    for layer_name in layer_names:
        layer = getattr(model, layer_name)
        handles.append(layer.fcm_module.register_forward_hook(make_hook(layer_name)))

    return handles, buffers, state


def collect_layer_features(
    model,
    protocol_file,
    dataset_name,
    split,
    layer_names,
    buffers,
    state,
    device,
    max_length,
    batch_size,
    num_workers,
    max_samples_per_class,
):
    print(f"\n[info] Collecting dataset: {dataset_name} ({protocol_file}), split={split}")

    for layer_name in layer_names:
        buffers[layer_name]["bonafide"].clear()
        buffers[layer_name]["spoof"].clear()

    dataset = DefaultDataset(prctl_path=protocol_file, transform=None, split=split, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        collate_fn=collate_fn,
    )

    collected = {name: {"bonafide": [], "spoof": []} for name in layer_names}
    counts = {name: {"bonafide": 0, "spoof": 0} for name in layer_names}

    with torch.no_grad():
        for audio, labels in tqdm(loader, desc=f"Collecting[{dataset_name}]", dynamic_ncols=True):
            audio = audio.type(torch.float32).to(device)
            labels = labels.view(-1).type(torch.int64).to(device)

            if hasattr(model, "proto_manager") and hasattr(model.proto_manager, "clear_labels"):
                model.proto_manager.clear_labels()
            force_clear_external_inits(model, layer_names)

            state["current_labels"] = labels
            _ = model(audio)

            if hasattr(model, "proto_manager") and hasattr(model.proto_manager, "clear_labels"):
                model.proto_manager.clear_labels()
            force_clear_external_inits(model, layer_names)

            for layer_name in layer_names:
                for class_name in ["bonafide", "spoof"]:
                    remaining = max_samples_per_class - counts[layer_name][class_name]
                    if remaining <= 0:
                        buffers[layer_name][class_name].clear()
                        continue

                    n_take = min(remaining, len(buffers[layer_name][class_name]))
                    if n_take > 0:
                        collected[layer_name][class_name].extend(buffers[layer_name][class_name][:n_take])
                        counts[layer_name][class_name] += n_take
                        del buffers[layer_name][class_name][:n_take]

            if all(
                counts[layer_name][class_name] >= max_samples_per_class
                for layer_name in layer_names
                for class_name in ["bonafide", "spoof"]
            ):
                break

    print("[info] Collected counts summary:")
    for layer_name in layer_names:
        print(
            f"  {layer_name}: "
            f"bona={counts[layer_name]['bonafide']}, "
            f"spoof={counts[layer_name]['spoof']}"
        )

    return collected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def align_cell_grid(ax, n_rows, n_cols, linewidth=0.6, alpha=0.18):
    ax.grid(False, which="both")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, n_cols, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color="k", linewidth=linewidth, alpha=alpha)

    ax.tick_params(which="minor", bottom=False, left=False)


def overlay_nan_hatch(ax, matrix, hatch="///", facecolor="lightgrey", alpha=0.35):
    nan_positions = np.argwhere(np.isnan(matrix))
    for row, col in nan_positions:
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=facecolor,
                edgecolor="none",
                hatch=hatch,
                linewidth=0.0,
                alpha=alpha,
            )
        )


def plot_sr_rank_heatmap_2x2(layer_names, layer_details, output_path, figure_title):
    """Plot the 2x2 SR-rank heatmap used in the high-order interaction analysis."""
    panels = [
        ("bonafide", "synergy_score_bonafide", "redundancy_score_bonafide"),
        ("spoof", "synergy_score_spoof", "redundancy_score_spoof"),
        ("Grad-Oinfo, y=spoof", "synergy_score_grad_y_spoof", "redundancy_score_grad_y_spoof"),
        ("Grad-Oinfo, y=bonafide", "synergy_score_grad_y_bonafide", "redundancy_score_grad_y_bonafide"),
    ]
    titles = ["Bona", "Spoof", "∇Ω | y=spoof", "∇Ω | y=bona"]

    layer_alias = {
        "HGNN_layer_S": "S",
        "HGNN_layer_T": "T",
        "HtrgHGNN_layer_ST11": "ST11",
        "HtrgHGNN_layer_ST12": "ST12",
        "HtrgHGNN_layer_ST21": "ST21",
        "HtrgHGNN_layer_ST22": "ST22",
    }

    max_slots = 0
    for _, synergy_key, _ in panels:
        for layer_name in layer_names:
            if layer_name in layer_details and synergy_key in layer_details[layer_name]:
                max_slots = max(max_slots, len(layer_details[layer_name][synergy_key]))

    if max_slots == 0:
        print("[warn] 2x2 heatmap skipped: no valid layer details.")
        return

    def build_matrix(synergy_key, redundancy_key):
        units = []
        synergy_values = []
        redundancy_values = []

        for layer_name in layer_names:
            details = layer_details.get(layer_name, None)
            if details is None or synergy_key not in details or redundancy_key not in details:
                continue

            synergy = np.asarray(details[synergy_key], dtype=np.float64).reshape(-1)
            redundancy = np.asarray(details[redundancy_key], dtype=np.float64).reshape(-1)
            n_slots = min(len(synergy), len(redundancy))

            for slot_idx in range(n_slots):
                units.append((layer_name, slot_idx))
                synergy_values.append(float(synergy[slot_idx]))
                redundancy_values.append(float(redundancy[slot_idx]))

        matrix = np.full((len(layer_names), max_slots), np.nan, dtype=np.float64)
        if len(units) == 0:
            return matrix

        rank_synergy = rankdata_average_ties(np.asarray(synergy_values, dtype=np.float64))
        rank_redundancy = rankdata_average_ties(np.asarray(redundancy_values, dtype=np.float64))
        sr_display = minmax_normalize(rank_synergy - rank_redundancy)

        layer_to_row = {layer_name: idx for idx, layer_name in enumerate(layer_names)}
        for idx, (layer_name, slot_idx) in enumerate(units):
            row = layer_to_row[layer_name]
            if slot_idx < max_slots:
                matrix[row, slot_idx] = sr_display[idx]

        return matrix

    matrices = [build_matrix(synergy_key, redundancy_key) for _, synergy_key, redundancy_key in panels]

    # Keep the layout settings of the original ideal visualization.
    try:
        fig, axs = plt.subplots(2, 2, figsize=(13.0, 7.6), dpi=200, layout="constrained")
    except TypeError:
        fig, axs = plt.subplots(2, 2, figsize=(13.0, 7.6), dpi=200, constrained_layout=True)

    if hasattr(fig, "get_layout_engine"):
        engine = fig.get_layout_engine()
        if engine is not None and hasattr(engine, "set"):
            engine.set(rect=(0.02, 0.02, 0.98, 0.90))

    axs = axs.ravel()

    step = 4 if max_slots >= 16 else 2
    xticks = list(range(0, max_slots, step))
    if (max_slots - 1) not in xticks:
        xticks.append(max_slots - 1)

    ytick_positions = np.arange(len(layer_names))
    ytick_labels = [layer_alias.get(name, name) for name in layer_names]

    images = []
    for panel_idx, (ax, title, matrix) in enumerate(zip(axs, titles, matrices)):
        image = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            cmap="RdBu_r",
        )
        images.append(image)

        align_cell_grid(ax, n_rows=matrix.shape[0], n_cols=matrix.shape[1], linewidth=0.6, alpha=0.18)
        overlay_nan_hatch(ax, matrix)

        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels, fontsize=16)

        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=11)

        ax.set_title(title, fontsize=16, fontweight="semibold", pad=2)

        row, col = divmod(panel_idx, 2)
        if col == 1:
            ax.tick_params(axis="y", labelleft=False)
        if row == 0:
            ax.tick_params(axis="x", labelbottom=False)

        ax.tick_params(length=0)

    colorbar = fig.colorbar(images[0], ax=axs.tolist(), fraction=0.025, pad=0.01)
    colorbar.set_label("SR Rank (0: Redundancy, 1: Synergy)", fontsize=12)
    colorbar.ax.tick_params(labelsize=12, length=0)

    fig.suptitle(figure_title, fontsize=24, fontweight="bold", y=0.995)
    fig.supxlabel("Slot idx", fontweight="bold", fontsize=24)
    fig.supylabel("Layer", fontweight="bold", fontsize=24)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[info] saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_dataset(collected_features, dataset_name, layer_names, hoi_method):
    rows = []
    layer_details = {}

    for layer_name in layer_names:
        bonafide_features = collected_features[layer_name]["bonafide"]
        spoof_features = collected_features[layer_name]["spoof"]

        if len(bonafide_features) == 0 or len(spoof_features) == 0:
            print(f"[warn] {dataset_name}/{layer_name}: insufficient samples; skipped.")
            continue

        X_bonafide = np.stack(bonafide_features, axis=0)
        X_spoof = np.stack(spoof_features, axis=0)
        num_variables = X_bonafide.shape[1]

        try:
            oinfo_bonafide, triplets_bonafide = compute_oinfo_triplets(X_bonafide, method=hoi_method)
            oinfo_spoof, triplets_spoof = compute_oinfo_triplets(X_spoof, method=hoi_method)
        except Exception as exc:
            print(f"[error] O-information failed for {dataset_name}/{layer_name}: {exc}")
            continue

        (
            layer_sr_bonafide,
            synergy_ratio_bonafide,
            synergy_score_bonafide,
            redundancy_score_bonafide,
            sr_rank_bonafide,
        ) = sr_rank_from_triplets(oinfo_bonafide, triplets_bonafide, num_variables)

        (
            layer_sr_spoof,
            synergy_ratio_spoof,
            synergy_score_spoof,
            redundancy_score_spoof,
            sr_rank_spoof,
        ) = sr_rank_from_triplets(oinfo_spoof, triplets_spoof, num_variables)

        grad_mean_y_spoof = np.nan
        grad_synergy_ratio_y_spoof = np.nan
        layer_sr_grad_y_spoof = np.nan
        synergy_score_grad_y_spoof = None
        redundancy_score_grad_y_spoof = None
        sr_rank_grad_y_spoof = None

        grad_mean_y_bonafide = np.nan
        grad_synergy_ratio_y_bonafide = np.nan
        layer_sr_grad_y_bonafide = np.nan
        synergy_score_grad_y_bonafide = None
        redundancy_score_grad_y_bonafide = None
        sr_rank_grad_y_bonafide = None

        X_all = np.concatenate([X_bonafide, X_spoof], axis=0)
        y_spoof = np.concatenate(
            [
                np.zeros((X_bonafide.shape[0],), dtype=np.float64),
                np.ones((X_spoof.shape[0],), dtype=np.float64),
            ],
            axis=0,
        )
        y_bonafide = 1.0 - y_spoof

        try:
            grad_spoof, triplets_grad_spoof = compute_gradient_oinfo_triplets(X_all, y_spoof, method=hoi_method)
            (
                layer_sr_grad_y_spoof,
                grad_synergy_ratio_y_spoof,
                synergy_score_grad_y_spoof,
                redundancy_score_grad_y_spoof,
                sr_rank_grad_y_spoof,
            ) = sr_rank_from_triplets(grad_spoof, triplets_grad_spoof, num_variables)
            grad_mean_y_spoof = float(np.mean(grad_spoof))
        except Exception as exc:
            print(f"[warn] GradientOinfo(y=spoof) failed for {dataset_name}/{layer_name}: {exc}")

        try:
            grad_bonafide, triplets_grad_bonafide = compute_gradient_oinfo_triplets(
                X_all, y_bonafide, method=hoi_method
            )
            (
                layer_sr_grad_y_bonafide,
                grad_synergy_ratio_y_bonafide,
                synergy_score_grad_y_bonafide,
                redundancy_score_grad_y_bonafide,
                sr_rank_grad_y_bonafide,
            ) = sr_rank_from_triplets(grad_bonafide, triplets_grad_bonafide, num_variables)
            grad_mean_y_bonafide = float(np.mean(grad_bonafide))
        except Exception as exc:
            print(f"[warn] GradientOinfo(y=bonafide) failed for {dataset_name}/{layer_name}: {exc}")

        rows.append(
            {
                "dataset": dataset_name,
                "layer": layer_name,
                "n_vars": int(num_variables),
                "global_similarity_bonafide_mean": float(np.mean(X_bonafide)),
                "global_similarity_spoof_mean": float(np.mean(X_spoof)),
                "oinfo_bonafide_mean": float(np.mean(oinfo_bonafide)),
                "oinfo_spoof_mean": float(np.mean(oinfo_spoof)),
                "oinfo_bonafide_synergy_ratio": float(synergy_ratio_bonafide),
                "oinfo_spoof_synergy_ratio": float(synergy_ratio_spoof),
                "layer_sr_bonafide": float(layer_sr_bonafide),
                "layer_sr_spoof": float(layer_sr_spoof),
                "grad_oinfo_mean_y_spoof": float(grad_mean_y_spoof),
                "grad_oinfo_synergy_ratio_y_spoof": float(grad_synergy_ratio_y_spoof),
                "layer_sr_grad_y_spoof": float(layer_sr_grad_y_spoof),
                "grad_oinfo_mean_y_bonafide": float(grad_mean_y_bonafide),
                "grad_oinfo_synergy_ratio_y_bonafide": float(grad_synergy_ratio_y_bonafide),
                "layer_sr_grad_y_bonafide": float(layer_sr_grad_y_bonafide),
            }
        )

        layer_details[layer_name] = {
            "layer_sr_bonafide": float(layer_sr_bonafide),
            "layer_sr_spoof": float(layer_sr_spoof),
            "sr_rank_bonafide": sr_rank_bonafide.tolist(),
            "sr_rank_spoof": sr_rank_spoof.tolist(),
            "synergy_score_bonafide": synergy_score_bonafide.tolist(),
            "redundancy_score_bonafide": redundancy_score_bonafide.tolist(),
            "synergy_score_spoof": synergy_score_spoof.tolist(),
            "redundancy_score_spoof": redundancy_score_spoof.tolist(),
        }

        if synergy_score_grad_y_spoof is not None:
            layer_details[layer_name]["layer_sr_grad_y_spoof"] = float(layer_sr_grad_y_spoof)
            layer_details[layer_name]["sr_rank_grad_y_spoof"] = sr_rank_grad_y_spoof.tolist()
            layer_details[layer_name]["synergy_score_grad_y_spoof"] = synergy_score_grad_y_spoof.tolist()
            layer_details[layer_name]["redundancy_score_grad_y_spoof"] = redundancy_score_grad_y_spoof.tolist()

        if synergy_score_grad_y_bonafide is not None:
            layer_details[layer_name]["layer_sr_grad_y_bonafide"] = float(layer_sr_grad_y_bonafide)
            layer_details[layer_name]["sr_rank_grad_y_bonafide"] = sr_rank_grad_y_bonafide.tolist()
            layer_details[layer_name]["synergy_score_grad_y_bonafide"] = synergy_score_grad_y_bonafide.tolist()
            layer_details[layer_name]["redundancy_score_grad_y_bonafide"] = redundancy_score_grad_y_bonafide.tolist()

    return rows, layer_details


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"[info] saved: {path}")


def write_summary_csv(path, rows):
    if len(rows) == 0:
        return

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[info] saved: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SR-rank heatmaps for HyperPotter high-order interaction analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=str,
        required=True,
        help="Path to the trained HyperPotter model checkpoint.",
    )
    parser.add_argument(
        "--prototype-banks-path",
        "--proto-banks-path",
        "--proto_banks_path",
        dest="prototype_banks_path",
        type=str,
        required=True,
        help="Path to the saved prototype banks.",
    )
    parser.add_argument(
        "--protocol-files",
        "--protocols-paths",
        "--protocols_paths",
        "--protocols-path",
        "--protocols_path",
        dest="protocol_files",
        type=str,
        nargs="+",
        required=True,
        help="One or more protocol files.",
    )
    parser.add_argument(
        "--dataset-names",
        "--dataset_names",
        dest="dataset_names",
        type=str,
        nargs="+",
        default=None,
        help="Dataset names used for output folders and figure titles.",
    )
    parser.add_argument(
        "--splits",
        dest="splits",
        type=str,
        nargs="+",
        default=None,
        help="Split name for each protocol file.",
    )
    parser.add_argument(
        "--split",
        dest="split",
        type=str,
        default="eval",
        help="Default split name used when --splits is not provided.",
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="./outputs/oinfo_visualization")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--max-length", "--max_length", dest="max_length", type=int, default=10000)
    parser.add_argument("--max-samples-per-class", "--max_samples_per_class", dest="max_samples_per_class", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bonafide-label", "--label_bonafide", dest="bonafide_label", type=int, default=1)
    parser.add_argument("--hoi-method", "--hoi_method", dest="hoi_method", type=str, default="gc")
    parser.add_argument("--layer-names", "--layer_keys", dest="layer_names", type=str, nargs="+", default=None)
    parser.add_argument(
        "--figure-title",
        dest="figure_title",
        type=str,
        default="SR-rank Heatmaps on Layer-wise Hypergraph Centroids",
        help="Global title of the 2x2 SR-rank heatmap.",
    )
    parser.add_argument(
        "--figure-formats",
        dest="figure_formats",
        type=str,
        nargs="+",
        default=["pdf", "png"],
        choices=["pdf", "png"],
        help="Figure formats to save.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def normalize_cli_lists(args):
    num_protocols = len(args.protocol_files)

    if args.dataset_names is None:
        args.dataset_names = [
            os.path.splitext(os.path.basename(path))[0] for path in args.protocol_files
        ]

    if args.splits is None:
        args.splits = [args.split] * num_protocols

    if not (len(args.protocol_files) == len(args.dataset_names) == len(args.splits)):
        raise ValueError("Lengths of --protocol-files, --dataset-names, and --splits must match.")

    return args


def main():
    args = normalize_cli_lists(parse_args())

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    model = build_model(device=device)
    model.to(device)
    model.eval()

    load_state_dict_flexible(model, args.model_path, device)
    load_prototype_banks(model, args.prototype_banks_path, device)

    if args.debug and hasattr(model, "proto_manager") and hasattr(model.proto_manager, "banks"):
        print("[debug] prototype bank keys:", sorted(model.proto_manager.banks.keys()))

    layer_names = discover_hypergraph_layers(model, preferred_layers=args.layer_names)
    if len(layer_names) == 0:
        raise RuntimeError("No layers with `fcm_module` found. Check model version or --layer-names.")
    print("[info] using layers:", layer_names)

    handles, buffers, state = register_similarity_hooks(
        model,
        layer_names=layer_names,
        bonafide_label=args.bonafide_label,
        debug=args.debug,
    )

    output_root = make_dir(args.output_dir)
    all_rows = []

    try:
        for protocol_file, dataset_name, split in zip(args.protocol_files, args.dataset_names, args.splits):
            dataset_output_dir = make_dir(os.path.join(output_root, dataset_name))

            collected_features = collect_layer_features(
                model=model,
                protocol_file=protocol_file,
                dataset_name=dataset_name,
                split=split,
                layer_names=layer_names,
                buffers=buffers,
                state=state,
                device=device,
                max_length=args.max_length,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples_per_class=args.max_samples_per_class,
            )

            rows, layer_details = analyze_dataset(
                collected_features=collected_features,
                dataset_name=dataset_name,
                layer_names=layer_names,
                hoi_method=args.hoi_method,
            )

            all_rows.extend(rows)

            write_json(os.path.join(dataset_output_dir, "summary.json"), rows)
            write_json(os.path.join(dataset_output_dir, "layer_details.json"), layer_details)

            for fmt in args.figure_formats:
                figure_path = os.path.join(dataset_output_dir, f"sr_rank_heatmap_2x2.{fmt}")
                plot_sr_rank_heatmap_2x2(
                    layer_names=layer_names,
                    layer_details=layer_details,
                    output_path=figure_path,
                    figure_title=args.figure_title,
                )

    finally:
        for handle in handles:
            handle.remove()

    write_summary_csv(os.path.join(output_root, "summary.csv"), all_rows)

    print(f"\n[done] Output dir: {output_root}")


if __name__ == "__main__":
    main()
