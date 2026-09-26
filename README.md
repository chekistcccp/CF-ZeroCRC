# CF-ZeroCRC

Tri-prompt counterfactual diffusion flow for **training-free colorectal tumor localization on CT**. This is a research prototype, not a medical device or a clinical decision tool. The frozen backbone is Stable Diffusion 3.5 Medium, downloaded from ModelScope.

The same noisy CT latent is evaluated with neutral (`P0`), healthy (`PH`), and cancer (`PA`) text. The primary response is `ReLU(||F(PH)-F(P0)||₂ - ||F(PA)-F(P0)||₂)`. See [the experiment design](docs/EXPERIMENT_DESIGN.md) for the hypothesis and required controls.

## WSL2 setup

Run from a WSL2 Linux shell in the repository root. Verify GPU access with `nvidia-smi`. A standard RTX 4090 has 24 GB VRAM. Record peak VRAM and runtime on one case before a full run; the default fine stage is expensive.

```bash
conda create -n cfzerocrc python=3.11 -y
conda activate cfzerocrc
# Install a CUDA-enabled PyTorch build appropriate for the WSL2 driver first.
pip install torch torchvision
pip install -r requirements.txt
pip install -e .
pip install pytest
PYTHONPATH=src python -m pytest -q
bash run.sh download
```

The model directory defaults to `models/sd35-medium`. Adjust `configs/default.yaml` if it is elsewhere. Large data and model files are excluded by `.gitignore`.

## Data layout

Place the original MSD Colon archive in `data/MSD/` as a single `.tar`, `.tar.gz`, or `.tgz` file. `bash run.sh prepare-msd` extracts it safely and finds its `imagesTr/` and `labelsTr/` directories, even when the archive contains an outer `Task10_Colon/` folder. MSD `imagesTr` and `labelsTr` are the public **labeled training set**; make a frozen development/evaluation split before tuning.

MSI files must have this structure:

```text
data/MSI/
├── data/
│   └── 10001_men_jing_mai(0.5mm_low)_20180808084217.nii.gz
└── data2/
    └── 10001.nii.gz
```

For each MSI image, the leading digits before its other filename characters must equal a pure-digit label filename. The software rejects missing or duplicate pairs. Labels are never sent to the model; they are read only during evaluation.

## Run

```bash
bash run.sh prepare-msd
bash run.sh msd
bash run.sh eval-msd
bash run.sh msi
bash run.sh eval-msi
```

`msd` runs exhaustive fine inference on valid body slices and writes `outputs/msd/`. `msi` uses coarse-to-fine inference and writes `outputs/msi/`. For fair MSI localization comparisons, run exhaustive mode on a predefined subset as well: coarse selection can miss an entire lesion.

Individual commands:

```bash
python scripts/run_inference.py --dataset msi --config configs/default.yaml --input data/MSI --output outputs/msi --mode coarse_fine
python scripts/run_inference.py --dataset msd --config configs/default.yaml --input data/MSD/Task10_Colon/imagesTr --output outputs/msd --mode exhaustive
python scripts/evaluate_msd.py --dataset msi --pred-dir outputs/msi --gt-dir data/MSI --output outputs/msi_metrics.json
```

The extracted MSD directory may differ from the example path; `run.sh` discovers it automatically. Every case generates a continuous NIfTI heatmap, a binary mask, per-slice scores, connected-component candidates, and a manifest. MSI output names use the numeric case ID.

## Evaluation and interpretation

Evaluation requires predictions for **every** label in the selected cohort. Labels are resampled to prediction space using NIfTI affines; a matching array shape alone is insufficient. AUROC and AUPRC are calculated inside a label-free body mask. Report macro case metrics and the number of evaluated cases. A missing prediction stops evaluation.

Coarse selection uses raw flow strength. In the fine stage, within-slice normalized fusion is weighted by raw flow strength before volume normalization to restore a slice-level amplitude signal. Its calibration across scans still needs empirical validation. The current 3D box IoU uses the union of predicted components. Implement lesion-level FROC with a fixed component matching and ranking rule before making a detection claim.

Do not tune prompts, noise levels, thresholds, or fusion weights on the final evaluation cases. If labeled development cases guide these settings, describe the method as **training-free**, not strictly annotation-free. Test normal colon cases, irrelevant-disease prompts, and prompt swaps to check whether the signal reflects lesions rather than text-conditioning strength. A cohort containing only cancer cases cannot establish screening specificity.

Stable Diffusion 3.5 Medium weights have their own upstream license; review it before downloading or redistributing weights.
