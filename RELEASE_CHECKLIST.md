# Source release checklist

Upload only source code, documentation, measured conductance data, and split indices. The official OCTMNIST NPZ is downloaded separately using the link and MD5 in `README.md`. Generated results remain local and are excluded by `.gitignore`.

## Files to upload

- `README.md`, `README_zh-CN.md`, `requirements.txt`, `RELEASE_CHECKLIST.md`, `.gitignore`
- `main.py`, `run_ablation.py`, `run_multiseed.py`, `export_validation_exclusions.py`, `reproduce_fig5_revised.py`, `plot_pulse.py`
- `line_LTP_LTD.csv`, `data/split_indices.csv`, `data/validation_excluded_indices.csv`

Before publication, verify that the listed files are complete, review data redistribution permissions, and record the source-code commit.
