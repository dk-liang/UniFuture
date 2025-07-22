import argparse
import torch
from tabulate import tabulate
import os
import sys
import os.path as osp

sys.path.append(os.getcwd())
from unifuture.modules.perception.depth_eval import metric
from unifuture.modules.perception.depth_eval.metric import DepthMetricTracker
from unifuture.modules.perception.depth_eval.eval import eval_depth


DEPTH_EVAL_METRICS = [
    "abs_relative_difference",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
]

def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--depth_dir",
        type=str,
        required=True
    )
    return parser


def load_and_concat_from_dir(depth_dir, num_files=8):
    preds = list()
    targets = list()
    
    for id in range(num_files):
        file_path = osp.join(depth_dir, f'cuda_{id}_depth.pt')
        if osp.exists(file_path):
            print(f'Now loading items in {file_path}')
            tensor_dict = torch.load(file_path, map_location='cpu')
            preds.extend([torch.mean(tensor, dim=1, keepdim=True) if tensor.shape[1] == 3 else tensor for tensor in tensor_dict['pred']])
            targets.extend([torch.mean(tensor, dim=1, keepdim=True) if tensor.shape[1] == 3 else tensor for tensor in tensor_dict['target']])
        else:
            print(f'Skip loading {file_path}')

    return preds, targets


def calc_metric(depth_preds, depth_targets, no_cuda=False):
    assert len(depth_preds) == len(depth_targets)
    print(f'Now calculating depth matrics, total sample num: {len(depth_preds)}')

    metric_funcs = [getattr(metric, _met) for _met in DEPTH_EVAL_METRICS]
    metric_tracker = DepthMetricTracker(*[m.__name__ for m in metric_funcs])
    metric_tracker.reset()  

    for pred, target in zip(depth_preds, depth_targets):
        eval_depth(pred, target, metric_funcs, metric_tracker, no_cuda=no_cuda)

    eval_text = f"Depth evaluation metrics:\n"
    eval_text += tabulate(
        [metric_tracker.result().keys(), metric_tracker.result().values()]
    )
    print(eval_text)


if __name__ == '__main__':
    opt = parse_args().parse_args()

    depth_preds, depth_targets = load_and_concat_from_dir(opt.depth_dir)
    calc_metric(depth_preds, depth_targets)