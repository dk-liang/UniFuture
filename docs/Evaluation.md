# Inference

Download official model:

```shell
huggingface-cli download H-EmbodVis/UniFuture --include="*safetensors" --repo-type model  --local-dir ./ckpts/
```

or use `bin_to_st.py` to convert your training result `path_to/pytorch_model.bin` to `ckpts/unifuture.safetensors`:

```shell
python tools/bin_to_st.py
```

Run the following script for inference:

  ```shell
  bash tools/sampling.sh
  ```
or infer on single GPU:

  ```shell
  mkdir sample_logs
  python tools/sample.py --version unifuture --fvd_dist --fid_dist --save_depth --height 320 --width 576 --rand_gen --sample_index=0 --num_samples=6000 --save outputs > sample_logs/inference.log
  ```

Furthermore, you can define the inference process through following arguments:

  - `--action`: The mode of provided ground truth actions. It could be "traj", "cmd", "steer", or "goal", or we perform action-free prediction by default.
  - `--n_rounds`: The number of sampling rounds, which determines the duration to predict. Avoid setting it too large to prevent tensor elements from exceeding INT_MAX.
  - `--low_vram`: Enable the low VRAM mode.
  - `--n_steps`: The number of DDIM sampling steps.

# Calculate metrics
After the inference is completed, obtain the evaluation results as following:

AbsRel and $\delta_1$

```shell
python tools/depth_eval_dist.py --depth_dir outputs/depth_saved_for_metric
```

FID

```shell
python tools/fid_eval_dist.py --feats_dir outputs/feats_saved_for_metric
```

FVD

```shell
mkdir sample_outputs && cd sample_outputs
ln -s ../outputs ./outputs && cd ..
python tools/eval_fvd/eval.py --yaml tools/eval_fvd/fvd.yaml
```