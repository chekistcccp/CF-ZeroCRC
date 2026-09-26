# CF-ZeroCRC

Tri-prompt counterfactual diffusion flow for **training-free colorectal tumor localization on CT**. This is a research prototype, not a medical device or a clinical decision tool. The frozen backbone is Stable Diffusion 3.5 Medium, downloaded from ModelScope.

The same noisy CT latent is evaluated with neutral (`P0`), healthy (`PH`), and cancer (`PA`) text. The primary response is `ReLU(||F(PH)-F(P0)||₂ - ||F(PA)-F(P0)||₂)`. See [the experiment design](docs/EXPERIMENT_DESIGN.md) for the hypothesis and required controls.

## WSL2 setup

Run from a WSL2 Linux shell in the repository root. Verify GPU access with `nvidia-smi` first. A standard RTX 4090 has 24 GB VRAM. The default fine stage is expensive.

```bash
bash run.sh
```

`run.sh` creates `.venv`, installs PyTorch and project dependencies, checks CUDA, prepares MSD, checks MSI image/label pairs, downloads SD3.5 Medium if absent, then performs both datasets' inference and evaluation. It stops on an error and can be run again; complete cases from the same configuration are skipped. Python 3 with `venv`, `pip`, and `sha256sum` must be available in WSL2. Use `PYTHON_BIN=/path/to/python3 bash run.sh` only if your Python executable is not `python3`.

The model directory defaults to `models/sd35-medium`. Adjust `configs/default.yaml` if it is elsewhere. Large data and model files are excluded by `.gitignore`.

## Data layout

Place the original MSD Colon archive in `data/MSD/` as a single `.tar`, `.tar.gz`, or `.tgz` file. The main script extracts it safely and finds its `imagesTr/` and `labelsTr/` directories, even when the archive contains an outer `Task10_Colon/` folder. MSD `imagesTr` and `labelsTr` are the public **labeled training set**; make a frozen development/evaluation split before tuning.

MSI files must have this structure:

```text
data/MSI/
├── data/
│   └── 10001_men_jing_mai(0.5mm_low)_20180808084217.nii.gz
└── data2/
    └── 10001.nii.gz
```

For each MSI image, the leading digits before its other filename characters must equal a pure-digit label filename. The software rejects missing or duplicate pairs. Labels are never sent to the model; they are read only during evaluation.

## Pipeline outputs

The automatic MSD run uses exhaustive fine inference on valid body slices and writes `outputs/msd/` and `outputs/msd_metrics.json`. The MSI run uses coarse-to-fine inference and writes `outputs/msi/` and `outputs/msi_metrics.json`. For fair MSI localization comparisons, run exhaustive mode on a predefined subset as well: coarse selection can miss an entire lesion.

The extracted MSD directory may differ from the example path; `run.sh` discovers it automatically. Every case generates a continuous NIfTI heatmap, a binary mask, per-slice scores, connected-component candidates, and a manifest. MSI output names use the numeric case ID.

## Evaluation and interpretation

Evaluation requires predictions for **every** label in the selected cohort. Labels are resampled to prediction space using NIfTI affines; a matching array shape alone is insufficient. AUROC and AUPRC are calculated inside a label-free body mask. Report macro case metrics and the number of evaluated cases. A missing prediction stops evaluation.

Coarse selection uses raw flow strength. In the fine stage, within-slice normalized fusion is weighted by raw flow strength before volume normalization to restore a slice-level amplitude signal. Its calibration across scans still needs empirical validation. The current 3D box IoU uses the union of predicted components. Implement lesion-level FROC with a fixed component matching and ranking rule before making a detection claim.

Do not tune prompts, noise levels, thresholds, or fusion weights on the final evaluation cases. If labeled development cases guide these settings, describe the method as **training-free**, not strictly annotation-free. Test normal colon cases, irrelevant-disease prompts, and prompt swaps to check whether the signal reflects lesions rather than text-conditioning strength. A cohort containing only cancer cases cannot establish screening specificity.

Stable Diffusion 3.5 Medium weights have their own upstream license; review it before downloading or redistributing weights.
