# Training

All training and inference processes are performed using $1\times8$ NVIDIA H20 GPUs (96 GB). We use DeepSpeed ZeRO stage 2 to improve data parallelism and lower memory footprint during training.

Multi-GPU training can be launched as:

  ```shell
  torchrun \
      --nnodes=1 \
      --nproc_per_node=8 \
      tools/train.py \
      --base configs/training.yaml \
      --finetune ckpts/vista.safetensors \
      --num_nodes 1 \
      --n_devices 8 
  ```

or through the training script:

  ```shell
  bash tools/training.sh
  ```

To debug on single GPU, run:

  ```shell
  python tools/train.py \
      --base configs/training.yaml \
      --finetune ckpts/vista.safetensors \
      --num_nodes 1 \
      --n_devices 1
  ```

> You can specify the directory to save these logs by providing an available path to `--logdir`.

The log directory contains a Python script named `zero_to_fp32.py` and a `checkpoint` folder that contains all partitioned checkpoints. The final checkpoint can be obtained by:

  ```shell
  python zero_to_fp32.py . pytorch_model.bin
  ```

---
Please refer to [Evaluation.md](./Evaluation.md) for instructions on evaluation.