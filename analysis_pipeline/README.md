# analysis_pipeline

Compute-only modules for bead tracking and autocorrelation.

## Goals
- Keep heavy per-frame computation in Python modules.
- Keep notebook focused on orchestration, QC, and plotting.
- Preserve the existing behavior from filament analysis notebooks while reducing duplication.

## Modules
- config.py: load YAML or JSON config and merge notebook/CLI overrides.
- io_dataset.py: load lazy TCZYX dataset and calibration, prepare output directories.
- beads_track.py: bead preview, frame-wise detections, track linking, parquet outputs.
- beads_velocity.py: velocity and angular speed computation from tracks.
- autocorr_3d.py: single-frame and sampled 3D autocorrelation with decay fitting.
- autocorr_2d.py: sampled and radial 2D autocorrelation with decay fitting.
- pipeline.py: high-level stage orchestrators used by notebook and CLI.
- cli.py: optional command-line entrypoint.

## Notebook Entry Point
Use analysis_unified.ipynb for running stages and plotting outputs.

## CLI Usage
Run all stages:

python -m analysis_pipeline.cli --config config/analysis_default.yaml --stage all

Run only bead stage:

python -m analysis_pipeline.cli --config config/analysis_default.yaml --stage beads

Run only autocorrelation stage:

python -m analysis_pipeline.cli --config config/analysis_default.yaml --stage autocorr

Force recompute:

python -m analysis_pipeline.cli --config config/analysis_default.yaml --stage all --force
