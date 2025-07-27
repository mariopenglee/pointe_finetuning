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

## Data pre-processing

```bash
python -m scripts.sample_pointclouds --output_dir my/custom/dir
```
## Configuration

Edit `config/config.yaml` to adjust data paths, model names, and hyperparameters.

## Finetuning

```bash
python -m finetune.train --config config/config.yaml
```

## Inference

```bash
python -m finetune.inference --config config/config.yaml --prompt "broken couch" --out broken_couch_pc.pt
```

## Visualizing Point Clouds

You can visualize point cloud `.pt` files using the provided script:

```bash
python scripts/visualize_pointclouds.py --dir data/pointclouds --n_show 5
```

- To visualize a specific point cloud file:

```bash
python scripts/visualize_pointclouds.py --dir data/pointclouds --file brokecouch_pc.pt
```

- By default, the script will show 10 random point clouds from the specified directory. You can change the number with `--n_show`.

## Sampling Point Clouds from Meshes

You can generate Point-E–style point clouds from your 3D mesh `.obj` files using the provided script:

```bash
python scripts/sample_pointclouds.py
```

- By default, this will look for `.obj` files in `data/objects/` and output point clouds to `data/pointclouds/`.
- Each mesh will be sampled to 1024 points (with color if available) and saved as a `.npz` file in the Point-E format (using `PointCloud.save`).

You can modify the input and output directories by editing the variables at the bottom of `scripts/sample_pointclouds.py`.

## Converting Point Clouds to Meshes

You can convert point clouds to meshes using the provided script:

```bash
python scripts/pointclouds_to_mesh.py --input data/pointclouds/brokecouch_pc.npz
```

- To convert all `.npz` point clouds in a directory:

```bash
python scripts/pointclouds_to_mesh.py --input data/pointclouds/
```

- The output mesh will be saved as a `.ply` file in the same directory by default, or you can specify an output directory with `--output`.
- The script uses the Point-E SDF model and supports GPU or CPU (set with `--device`).

> **Note:** The script expects point clouds in `.npz` format as produced by `PointCloud.save()`. If your point clouds are in `.pt` format, you may need to convert them first.
