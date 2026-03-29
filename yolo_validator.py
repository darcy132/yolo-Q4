#!/usr/bin/env python3
"""
YOLO Dataset Validator
验证 YOLO 格式数据集的完整性和正确性
"""

import os
import sys
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import argparse

# ── ANSI colors ──────────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def ok(msg):   print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"  {RED}✘{RESET}  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
class DatasetValidator:
    def __init__(self, dataset_root: str):
        self.root   = Path(dataset_root).resolve()
        self.errors = 0
        self.warns  = 0

    # ── helpers ───────────────────────────────────────────────────────────────
    def _err(self, msg):
        err(msg); self.errors += 1

    def _warn(self, msg):
        warn(msg); self.warns += 1

    # ── 1. directory structure ────────────────────────────────────────────────
    def check_structure(self):
        section("1. Directory Structure")
        required = [
            "dataset.yaml",
            "classes.txt",
            "images/train",
            "images/val",
            "labels/train",
            "labels/val",
        ]
        for rel in required:
            p = self.root / rel
            if p.exists():
                ok(rel)
            else:
                self._err(f"Missing: {rel}")

    # ── 2. dataset.yaml ───────────────────────────────────────────────────────
    def check_yaml(self) -> dict:
        section("2. dataset.yaml")
        yaml_path = self.root / "dataset.yaml"
        if not yaml_path.exists():
            self._err("dataset.yaml not found — skipping yaml checks")
            return {}

        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)

        for key in ("nc", "names"):
            if key not in cfg:
                self._err(f"dataset.yaml missing key: '{key}'")
            else:
                ok(f"'{key}' present")

        if "nc" in cfg and "names" in cfg:
            nc    = cfg["nc"]
            names = cfg["names"]
            if len(names) != nc:
                self._err(f"nc={nc} but len(names)={len(names)}")
            else:
                ok(f"nc={nc} matches len(names)={len(names)}")

        # optional path keys
        for key in ("path", "train", "val"):
            if key in cfg:
                ok(f"'{key}': {cfg[key]}")
            else:
                self._warn(f"dataset.yaml has no '{key}' key")

        return cfg

    # ── 3. classes.txt ────────────────────────────────────────────────────────
    def check_classes(self, yaml_cfg: dict) -> list:
        section("3. classes.txt")
        cls_path = self.root / "classes.txt"
        if not cls_path.exists():
            self._err("classes.txt not found")
            return []

        classes = [l.strip() for l in cls_path.read_text().splitlines() if l.strip()]
        ok(f"{len(classes)} classes: {classes}")

        if yaml_cfg and "names" in yaml_cfg:
            if classes == list(yaml_cfg["names"]):
                ok("classes.txt matches dataset.yaml names")
            else:
                self._warn(
                    f"classes.txt != dataset.yaml names\n"
                    f"     classes.txt : {classes}\n"
                    f"     yaml names  : {list(yaml_cfg['names'])}"
                )
        return classes

    # ── 4. image / label pairing per split ───────────────────────────────────
    def check_split(self, split: str, num_classes: int) -> dict:
        section(f"4. Split: {split}")
        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split

        if not img_dir.exists() or not lbl_dir.exists():
            self._err(f"Directories missing for split '{split}'")
            return {}

        # collect stems
        img_stems = {
            p.stem: p
            for p in img_dir.iterdir()
            if p.suffix.lower() in VALID_IMG_EXTS
        }
        lbl_stems = {
            p.stem: p
            for p in lbl_dir.iterdir()
            if p.suffix == ".txt"
        }

        ok(f"Images : {len(img_stems)}")
        ok(f"Labels : {len(lbl_stems)}")

        # pairing
        imgs_no_label = set(img_stems) - set(lbl_stems)
        lbls_no_image = set(lbl_stems) - set(img_stems)

        if imgs_no_label:
            self._warn(f"{len(imgs_no_label)} image(s) with no label file")
            for s in sorted(imgs_no_label)[:5]:
                print(f"       {img_stems[s].name}")
            if len(imgs_no_label) > 5:
                print(f"       … and {len(imgs_no_label)-5} more")

        if lbls_no_image:
            self._warn(f"{len(lbls_no_image)} label file(s) with no image")
            for s in sorted(lbls_no_image)[:5]:
                print(f"       {lbl_stems[s].name}")
            if len(lbls_no_image) > 5:
                print(f"       … and {len(lbls_no_image)-5} more")

        if not imgs_no_label and not lbls_no_image:
            ok("Perfect 1-to-1 image ↔ label pairing")

        # ── label content validation ──────────────────────────────────────────
        stats = {
            "total_boxes": 0,
            "class_counts": Counter(),
            "bad_files": [],
            "empty_labels": [],
        }

        paired = set(img_stems) & set(lbl_stems)
        for stem in sorted(paired):
            lbl_path = lbl_stems[stem]
            lines    = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]

            if not lines:
                stats["empty_labels"].append(lbl_path.name)
                continue

            for i, line in enumerate(lines, 1):
                parts = line.split()
                # ── field count ───────────────────────────────────────────────
                if len(parts) < 5:
                    self._err(
                        f"{lbl_path.name} line {i}: expected ≥5 fields, got {len(parts)}"
                    )
                    stats["bad_files"].append(lbl_path.name)
                    continue

                # extra columns (e.g. segmentation masks) are allowed
                try:
                    cls_id = int(parts[0])
                    coords = [float(v) for v in parts[1:5]]
                except ValueError:
                    self._err(f"{lbl_path.name} line {i}: non-numeric value")
                    stats["bad_files"].append(lbl_path.name)
                    continue

                # ── class id range ────────────────────────────────────────────
                if cls_id < 0 or (num_classes > 0 and cls_id >= num_classes):
                    self._err(
                        f"{lbl_path.name} line {i}: class_id={cls_id} "
                        f"out of range [0, {num_classes-1}]"
                    )
                    stats["bad_files"].append(lbl_path.name)

                # ── bbox range [0, 1] ─────────────────────────────────────────
                x_c, y_c, w, h = coords
                if not (0.0 <= x_c <= 1.0 and 0.0 <= y_c <= 1.0 and
                        0.0 <  w  <= 1.0 and 0.0 <  h  <= 1.0):
                    self._err(
                        f"{lbl_path.name} line {i}: bbox out of [0,1] "
                        f"({x_c:.4f}, {y_c:.4f}, {w:.4f}, {h:.4f})"
                    )
                    stats["bad_files"].append(lbl_path.name)

                stats["total_boxes"]      += 1
                stats["class_counts"][cls_id] += 1

        # ── per-split summary ─────────────────────────────────────────────────
        ok(f"Total boxes     : {stats['total_boxes']}")
        ok(f"Avg boxes/image : {stats['total_boxes'] / max(len(paired), 1):.1f}")

        if stats["empty_labels"]:
            self._warn(f"{len(stats['empty_labels'])} empty label file(s) (background images)")
            for f in stats["empty_labels"][:3]:
                print(f"       {f}")

        if stats["bad_files"]:
            n = len(set(stats["bad_files"]))
            self._err(f"{n} label file(s) with format errors (see above)")

        print(f"\n  {'Class ID':<10} {'Count':>8}   Bar")
        for cls_id in sorted(stats["class_counts"]):
            count = stats["class_counts"][cls_id]
            bar   = "█" * min(40, count * 40 // max(stats["class_counts"].values()))
            print(f"  {cls_id:<10} {count:>8}   {bar}")

        return stats

    # ── 5. train / val class distribution ────────────────────────────────────
    def check_distribution(self, train_stats: dict, val_stats: dict, classes: list):
        section("5. Train / Val Class Distribution")
        if not train_stats or not val_stats:
            self._warn("Cannot compare — one split had errors")
            return

        tc = train_stats["class_counts"]
        vc = val_stats["class_counts"]
        all_ids = sorted(set(tc) | set(vc))

        total_t = sum(tc.values()) or 1
        total_v = sum(vc.values()) or 1

        print(f"  {'ID':<4} {'Name':<20} {'Train%':>8} {'Val%':>8}  {'Δ%':>6}")
        print(f"  {'─'*4} {'─'*20} {'─'*8} {'─'*8}  {'─'*6}")
        for cid in all_ids:
            name  = classes[cid] if cid < len(classes) else f"cls_{cid}"
            tp    = tc[cid] / total_t * 100
            vp    = vc[cid] / total_v * 100
            delta = abs(tp - vp)
            flag  = f" {YELLOW}⚠{RESET}" if delta > 10 else ""
            print(f"  {cid:<4} {name:<20} {tp:>7.1f}% {vp:>7.1f}%  {delta:>5.1f}%{flag}")

        # missing classes
        only_train = set(tc) - set(vc)
        only_val   = set(vc) - set(tc)
        if only_train:
            self._warn(f"Classes in train but NOT in val: {sorted(only_train)}")
        if only_val:
            self._warn(f"Classes in val but NOT in train: {sorted(only_val)}")
        if not only_train and not only_val:
            ok("All classes present in both splits")

    # ── main entry ────────────────────────────────────────────────────────────
    def run(self):
        print(f"\n{BOLD}YOLO Dataset Validator{RESET}")
        print(f"Root: {self.root}")

        self.check_structure()
        yaml_cfg    = self.check_yaml()
        classes     = self.check_classes(yaml_cfg)
        num_classes = len(classes) if classes else (yaml_cfg.get("nc", 0) if yaml_cfg else 0)

        train_stats = self.check_split("train", num_classes)
        val_stats   = self.check_split("val",   num_classes)
        self.check_distribution(train_stats, val_stats, classes)

        # ── final verdict ─────────────────────────────────────────────────────
        section("Summary")
        if self.errors == 0 and self.warns == 0:
            print(f"  {GREEN}{BOLD}All checks passed — dataset looks healthy ✔{RESET}")
        elif self.errors == 0:
            print(f"  {YELLOW}{BOLD}{self.warns} warning(s), 0 errors — review warnings above{RESET}")
        else:
            print(f"  {RED}{BOLD}{self.errors} error(s), {self.warns} warning(s) — fix errors before training{RESET}")

        print()
        return self.errors


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a YOLO dataset")
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="dataset",
        help="Path to dataset root (default: ./dataset)",
    )
    args = parser.parse_args()

    validator = DatasetValidator(args.dataset_root)
    sys.exit(validator.run())