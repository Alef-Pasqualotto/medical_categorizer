# medical_categorizer
This repository aims to provide a clear and reproductible instance in order to run the experiments regarding the BERTimbau and the SaudeBR-QA dataset.

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA 12.4 ([check yours](#check-your-cuda-version))

---

## Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate jupyter_pln
```

### Option B — pip

First install PyTorch with the right CUDA build for your system (see [below](#different-cuda-version)), then the rest:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Different CUDA version?

### Check your CUDA version

```bash
nvidia-smi
```

Look for `CUDA Version` in the top-right of the output. Then pick the matching install:

| CUDA | PyTorch install |
|------|----------------|
| 12.4 | `--index-url https://download.pytorch.org/whl/cu124` |
| 12.1 | `--index-url https://download.pytorch.org/whl/cu121` |
| 11.8 | `--index-url https://download.pytorch.org/whl/cu118` |

**Conda users** — edit `environment.yml` before creating the env, changing the last line of the pytorch dependencies:

```yaml
- pytorch::pytorch-cuda=12.4   # change to 12.1 or 11.8
```

**pip users** — swap the `--index-url` in the torch install command above.

All available PyTorch builds: https://pytorch.org/get-started/previous-versions/

---

## Verify the installation

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## Running the experiment

```bash
python trainer.py  # replace with your entry point
```

---