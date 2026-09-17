# CF-ZeroCRC

**Tri-prompt Counterfactual Diffusion Flow for training-free colorectal tumor localization on CT.**

This repository implements a research prototype based on **Stable Diffusion 3.5 Medium** for zero-shot / annotation-free abnormal-region localization in colorectal CT. The model weights are downloaded from **ModelScope** and inference is designed for a single **RTX 4090 48GB** GPU in BF16.

> Research code only. This repository is not a medical device and must not be used for diagnosis or clinical decision-making.

## Method at a glance

For the same noisy CT latent `z_t`, SD3.5 receives three prompts:

- `P0`: neutral CT description
- `PH`: healthy / lesion-removed counterfactual
- `PA`: colorectal-cancer counterfactual

The core flow anomaly response is:

```text
D_H = ||F(z_t, PH) - F(z_t, P0)||_2
D_A = ||F(z_t, PA) - F(z_t, P0)||_2
M_CF = ReLU(D_H - D_A)
```

The fine stage optionally fuses:

1. multi-noise counterfactual flow maps;
2. paired counterfactual reconstruction residuals;
3. differential local-SSIM maps;
4. multi-seed consistency;
5. z-axis continuity.

See [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) for the full experimental design.

## Repository layout

```text
CF-ZeroCRC/
├── configs/default.yaml
├── docs/EXPERIMENT_DESIGN.md
├── scripts/
│   ├── download_model.py
│   ├── run_inference.py
│   └── evaluate_msd.py
├── src/cfzerocrc/
│   ├── io.py
│   ├── model.py
│   ├── pipeline.py
│   └── postprocess.py
├── tests/
├── requirements.txt
├── pyproject.toml
└── run.sh
```

## 1. Environment

Create the Conda environment manually, then install dependencies:

```bash
conda create -n cfzerocrc python=3.11 -y
conda activate cfzerocrc

# Install the CUDA build of PyTorch appropriate for your machine first.
# Example only; choose the command recommended by pytorch.org for your CUDA setup.
pip install torch torchvision

pip install -r requirements.txt
pip install -e .
```

The code expects CUDA and is tuned for a 48GB 4090. CPU execution is not a practical target for SD3.5.

## 2. Download SD3.5 Medium from ModelScope

```bash
python scripts/download_model.py \
  --model-id stabilityai/stable-diffusion-3.5-medium \
  --output models/sd35-medium
```

The downloader requests only the Diffusers-format directories needed by this project and skips the duplicate standalone checkpoint when ModelScope pattern filtering is available.

## 3. Prepare data

### Private CT

Put `.nii` / `.nii.gz` volumes under a directory, for example:

```text
data/private/
├── patient_0001.nii.gz
├── patient_0002.nii.gz
└── ...
```

No clinical variables are needed.

### MSD Task10 Colon (recommended for quantitative evaluation)

Expected layout:

```text
data/Task10_Colon/
├── imagesTr/
│   ├── colon_001.nii.gz
│   └── ...
└── labelsTr/
    ├── colon_001.nii.gz
    └── ...
```

## 4. Run inference

Fast coarse-to-fine mode for the private cohort:

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --input data/private \
  --output outputs/private \
  --mode coarse_fine
```

Exhaustive fine inference for a small quantitative set:

```bash
python scripts/run_inference.py \
  --config configs/default.yaml \
  --input data/Task10_Colon/imagesTr \
  --output outputs/msd \
  --mode exhaustive
```

Outputs per case include:

- continuous 3D anomaly heatmap (`*_heatmap.nii.gz`);
- thresholded candidate mask (`*_mask.nii.gz`);
- slice-level scores (`*_slices.csv`);
- connected-component candidates (`*_candidates.json`).

## 5. Evaluate on MSD Colon

```bash
python scripts/evaluate_msd.py \
  --pred-dir outputs/msd \
  --gt-dir data/Task10_Colon/labelsTr \
  --output outputs/msd_metrics.json
```

Implemented metrics:

- pixel/voxel AUROC;
- AUPRC;
- Dice after unsupervised thresholding;
- pointing-game accuracy;
- 3D bounding-box IoU.

For publication-quality experiments, report FROC as an additional detection metric after defining a fixed connected-component protocol.

## 6. Convenience launcher

```bash
bash run.sh download
bash run.sh infer data/private outputs/private
bash run.sh exhaustive data/Task10_Colon/imagesTr outputs/msd
bash run.sh eval outputs/msd data/Task10_Colon/labelsTr outputs/msd_metrics.json
```

## Important experimental rule

Do **not** tune thresholds, prompt wording, noise levels, or fusion weights on the final test set. Use a small development subset or fully label-free defaults, then freeze them before evaluation.

## Model license

Stable Diffusion 3.5 Medium is distributed under Stability AI's model license. Review the upstream license before downloading or redistributing weights. Model weights are intentionally excluded from this repository.
