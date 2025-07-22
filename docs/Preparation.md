# Preparation

This project uses PyTorch 2.0.1, CUDA 11.8, and recommends Conda for environment management.

## 1. Code and Conda Environment

Install code.
```shell
git clone https://github.com/dk-liang/UniFuture.git
cd UniFuture
```

Create the environment.

```shell
conda create -n unifuture python=3.9 -y
conda activate unifuture
```

Install dependencies.

```shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
pip3 install -e git+https://github.com/Stability-AI/datapipelines.git@main#egg=sdata
```

## 2. Data and Pretrained Models
Create an `annos` directory and put the JSON files downloaded [here](https://drive.google.com/drive/folders/1JpZObdR0OXagCbnPZfMSI8vhGLom5pht?usp=sharing) into it:

```shell
mkdir annos
cp /path/to/nuScenes.json ./annos/
cp /path/to/nuScenes_val.json ./annos/
```

Create an `data` directory, place or symlink your nuScenes dataset to it:

```shell
mkdir data
ln -s /path/to/your/nuscenes ./data/nuscenes
```

For training, create an `ckpts` directory and download pretrrained models into it:

```shell
mkdir ckpts
huggingface-cli download depth-anything/Depth-Anything-V2-Large --include "depth_anything_v2_vitl.pth"  --repo-type model --local-dir ./ckpts/
huggingface-cli download apple/DFN5B-CLIP-ViT-H-14 --include "open_clip_pytorch_model.bin" --repo-type model --local-dir ./ckpts/
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt --include "svd_xt.safetensors" --repo-type model --local-dir ./ckpts/
huggingface-cli download OpenDriveLab/Vista --include "vista.safetensors" --repo-type model --local-dir ./ckpts/
```

## Directory Structure

Your project directory should look like this after setup:

```
UniFuture
├── annos
|   ├── nuScenes.json
|   └── nuScenes_val.json
├── ckpts
|   ├── depth_anything_v2_vitl.pth
|   ├── open_clip_pytorch_model.bin
|   ├── svd_xt.safetensors
|   └── vista.safetensors
├── data
|   └── nuscenes
|       ├── samples
|       ├── sweeps
|       ├── ...
|       └── v1.0-trainval
...
```

---
Please refer to [Training.md](./Training.md) for instructions on training.