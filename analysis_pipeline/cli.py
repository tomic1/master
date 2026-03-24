from __future__ import annotations

import argparse

from .config import load_analysis_config, merge_overrides
from .pipeline import run_autocorr_core, run_bead_core, run_core_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run bead/autocorr analysis pipeline")
    p.add_argument("--config", required=True, help="Path to YAML/JSON config")
    p.add_argument(
        "--stage",
        choices=["beads", "autocorr", "all"],
        default="all",
        help="Pipeline stage to run",
    )
    p.add_argument("--dataset-id", default=None, help="Override dataset.dataset_id")
    p.add_argument("--base-dir", default=None, help="Override dataset.base_dir")
    p.add_argument("--force", action="store_true", help="Force recompute even if outputs exist")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_analysis_config(args.config)

    overrides = {}
    if args.dataset_id or args.base_dir:
        overrides["dataset"] = {}
        if args.dataset_id:
            overrides["dataset"]["dataset_id"] = args.dataset_id
        if args.base_dir:
            overrides["dataset"]["base_dir"] = args.base_dir

    if args.force:
        overrides.setdefault("runtime", {})["skip_existing"] = False

    final_config = merge_overrides(config, overrides)

    if args.stage == "beads":
        run_bead_core(final_config)
    elif args.stage == "autocorr":
        run_autocorr_core(final_config)
    else:
        run_core_pipeline(final_config)


if __name__ == "__main__":
    main()
