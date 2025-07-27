# nooneisthere
## a codebase for finetuning the text-to-clouds point-E model.
This repository provides a structured codebase for finetuning and inference of Point-E models.

## Setup

### Prerequisites

- Linux or macOS (or WSL on Windows)
- Python 3.8 or higher
- NVIDIA GPU with CUDA support (CUDA 11.1+)
- NVIDIA driver matching CUDA version
- Optional: [Conda](https://docs.conda.io/) for environment management

### Environment

#### Using Conda

```bash
conda create -n pointe python=3.8 -y
conda activate pointe
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
````

#### Using virtualenv

```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
```

## Installation

```bash
# Clone the Point-E repository
git clone https://github.com/openai/point-e.git

# Install Point-E in editable mode
pip install -e point-e

# Install training package requirements
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to adjust data paths, model names, and hyperparameters.

## Finetuning

```bash
python -m pointe_finetune.train --config config/config.yaml
```

## Inference

```bash
python -m pointe_finetune.inference --config config/config.yaml --prompt "broken couch" --out broken_couch_pc.pt
```
