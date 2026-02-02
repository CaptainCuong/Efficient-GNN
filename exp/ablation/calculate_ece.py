#!/usr/bin/env python3
"""
Calculate ECE for attacked nodes and draw reliability diagrams using ece_chart.
Uses probability arrays directly from JSON log file.
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import utils.ece as ece


def main():
    # Load JSON log file
    json_path = "./logs/ugca_rerank_basic_citeseer_ets_attack_log_20260201_213338.json"

    with open(json_path, 'r') as f:
        log_data = json.load(f)

    # Extract data from log
    detailed_results = log_data["detailed_results"]
    num_classes = log_data["experiment_info"]["config"]["num_classes"]

    # Get probabilities and labels from log
    original_probs = np.array([r["original_probs"] for r in detailed_results])
    attacked_probs = np.array([r["attacked_probs"] for r in detailed_results])
    true_labels = np.array([r["true_label"] for r in detailed_results])

    print("=" * 60)
    print("ECE CALCULATION FROM LOG FILE")
    print("=" * 60)
    print(f"JSON file: {json_path}")
    print(f"Number of attacked nodes: {len(detailed_results)}")
    print(f"Number of classes: {num_classes}")

    # Create images folder
    images_dir = "./images"
    os.makedirs(images_dir, exist_ok=True)

    # Draw ECE chart for all classes (original)
    print("\n" + "=" * 60)
    print("DRAWING ECE CHARTS")
    print("=" * 60)

    print("\nDrawing ECE chart for original (before attack)...")
    ece.ece_chart(
        original_probs,
        true_labels,
        n_classes=num_classes,
        n_bins=10,
        fig_name=os.path.join(images_dir, "ece_chart_original.png")
    )
    print(f"Saved: {os.path.join(images_dir, 'ece_chart_original.png')}")

    print("\nDrawing ECE chart for attacked (after attack)...")
    ece.ece_chart(
        attacked_probs,
        true_labels,
        n_classes=num_classes,
        n_bins=10,
        fig_name=os.path.join(images_dir, "ece_chart_attacked.png")
    )
    print(f"Saved: {os.path.join(images_dir, 'ece_chart_attacked.png')}")

    # Calculate and print ECE values
    print("\n" + "=" * 60)
    print("ECE VALUES")
    print("=" * 60)

    original_avg_ece = ece.calculate_average_ece(original_probs, true_labels, num_classes, logits=False)
    attacked_avg_ece = ece.calculate_average_ece(attacked_probs, true_labels, num_classes, logits=False)

    print(f"\nOriginal (before attack):")
    print(f"  Average ECE: {original_avg_ece:.4f}")
    for class_idx in range(num_classes):
        class_ece = ece.calculate_ece(original_probs, true_labels, class_idx, logits=False)
        print(f"  Class {class_idx} ECE: {class_ece:.4f}")

    print(f"\nAttacked (after attack):")
    print(f"  Average ECE: {attacked_avg_ece:.4f}")
    for class_idx in range(num_classes):
        class_ece = ece.calculate_ece(attacked_probs, true_labels, class_idx, logits=False)
        print(f"  Class {class_idx} ECE: {class_ece:.4f}")

    print(f"\nECE Change: {attacked_avg_ece - original_avg_ece:+.4f}")
    print(f"\nAll charts saved to: {images_dir}/")


if __name__ == "__main__":
    main()
