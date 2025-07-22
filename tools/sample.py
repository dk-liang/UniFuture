import argparse
import json
import random
from tabulate import tabulate
import logging

from pytorch_lightning import seed_everything
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

import init_proj_path
from sample_utils import *
from unifuture.modules.perception.depth_utils import get_depth_vis
from unifuture.modules.perception.depth_eval import metric
from unifuture.modules.perception.depth_eval.metric import DepthMetricTracker
from unifuture.modules.perception.depth_eval.eval import eval_depth
from unifuture.fvd_utils.fvd_utils import get_fvd_logits, frechet_distance, load_fvd_model, polynomial_mmd
from unifuture.fvd_utils.fid_utils import FlexibleFrechetInceptionDistance


VERSION2SPECS = {
    "unifuture": {
        "config": "configs/inference.yaml",
        "ckpt": "ckpts/unifuture.safetensors"
    }
}

DATASET2SOURCES = {
    "NUSCENES": {
        "data_root": "data/nuscenes",
        "anno_file": "annos/nuScenes_val.json"
    },
    "IMG": {
        "data_root": "image_folder"
    }
}

DEPTH_EVAL_METRICS = [
    "abs_relative_difference",
    "delta1_acc",
    "delta2_acc",
    "delta3_acc",
]


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--version",
        type=str,
        default="unifuture",
        help="model version"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NUSCENES",
        help="dataset name"
    )
    parser.add_argument(
        "--save",
        type=str,
        default="outputs",
        help="directory to save samples"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="free",
        help="action mode for control, such as traj, cmd, steer, goal"
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=1,
        help="number of sampling rounds"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=25,
        help="number of frames for each round"
    )
    parser.add_argument(
        "--n_conds",
        type=int,
        default=1,
        help="number of initial condition frames for the first round"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="random seed for seed_everything"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="target height of the generated video"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="target width of the generated video"
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="scale of the classifier-free guidance"
    )
    parser.add_argument(
        "--cond_aug",
        type=float,
        default=0.0,
        help="strength of the noise augmentation"
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=50,
        help="number of sampling steps"
    )
    parser.add_argument(
        "--rand_gen",
        action="store_false",
        help="whether to generate samples randomly or sequentially"
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="whether to save memory or not"
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="whether to calculuate the FID of the model"
    )
    parser.add_argument(
        "--fid_dist",
        action='store_true',
        help="whether to calculuate the FID of the model in distributed mode"
    )
    parser.add_argument(
        "--fvd",
        action="store_true",
        help="whether to calculuate the FVD of the model in CPU mode"
    )
    parser.add_argument(
        "--fvd_dist",
        action='store_true',
        help="whether to calculuate the FVD of the model in CPU mode in distributed mode"
    )
    parser.add_argument(
        "--fvd_cuda",
        action='store_true',
        help="whether to calculuate the FVD of the model in CUDA mode"
    )
    parser.add_argument(
        "--fvd_version",
        default='v1',
        help='the version of FVD used for evaluation, v1: from LVDM, v2: from Vista official'
    )
    parser.add_argument(
        "--depth_eval",
        action='store_true',
        help='whether to evaluate the depth branch'
    )
    parser.add_argument(
        "--save_depth",
        action='store_true',
        help="whether to save depth prediction and target (for evaluation)"
    )
    parser.add_argument(
        "--sample_index",
        default=0,
        type=int,
        help="the begining of sample index"
    )
    parser.add_argument(
        "--num_samples",
        default=None,
        type=int,
        help="the number of samples to generate"
    )
    return parser


def get_sample(selected_index=0, dataset_name="NUSCENES", num_frames=25, action_mode="free"):
    dataset_dict = DATASET2SOURCES[dataset_name]
    action_dict = None
    if dataset_name == "IMG":
        image_list = os.listdir(dataset_dict["data_root"])
        total_length = len(image_list)
        while selected_index >= total_length:
            selected_index -= total_length
        image_file = image_list[selected_index]

        path_list = [os.path.join(dataset_dict["data_root"], image_file)] * num_frames
    else:
        with open(dataset_dict["anno_file"], "r") as anno_json:
            all_samples = json.load(anno_json)
        total_length = len(all_samples)
        while selected_index >= total_length:
            selected_index -= total_length
        sample_dict = all_samples[selected_index]

        path_list = list()
        if dataset_name == "NUSCENES":
            for index in range(num_frames):
                image_path = os.path.join(dataset_dict["data_root"], sample_dict["frames"][index])
                assert os.path.exists(image_path), image_path
                path_list.append(image_path)
            if action_mode != "free":
                action_dict = dict()
                if action_mode == "traj" or action_mode == "trajectory":
                    action_dict["trajectory"] = torch.tensor(sample_dict["traj"][2:])
                elif action_mode == "cmd" or action_mode == "command":
                    action_dict["command"] = torch.tensor(sample_dict["cmd"])
                elif action_mode == "steer":
                    # scene might be empty
                    if sample_dict["speed"]:
                        action_dict["speed"] = torch.tensor(sample_dict["speed"][1:])
                    # scene might be empty
                    if sample_dict["angle"]:
                        action_dict["angle"] = torch.tensor(sample_dict["angle"][1:]) / 780
                elif action_mode == "goal":
                    # point might be invalid
                    if sample_dict["z"] > 0 and 0 < sample_dict["goal"][0] < 1600 and 0 < sample_dict["goal"][1] < 900:
                        action_dict["goal"] = torch.tensor([
                            sample_dict["goal"][0] / 1600,
                            sample_dict["goal"][1] / 900
                        ])
                else:
                    raise ValueError(f"Unsupported action mode {action_mode}")
        else:
            raise ValueError(f"Invalid dataset {dataset_name}")
    return path_list, selected_index, total_length, action_dict


def load_img(file_name, target_height=320, target_width=576, device="cuda"):
    if file_name is not None:
        image = Image.open(file_name)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    else:
        raise ValueError(f"Invalid image file {file_name}")
    ori_w, ori_h = image.size

    if ori_w / ori_h > target_width / target_height:
        tmp_w = int(target_width / target_height * ori_h)
        left = (ori_w - tmp_w) // 2
        right = (ori_w + tmp_w) // 2
        image = image.crop((left, 0, right, ori_h))
    elif ori_w / ori_h < target_width / target_height:
        tmp_h = int(target_height / target_width * ori_w)
        top = (ori_h - tmp_h) // 2
        bottom = (ori_h + tmp_h) // 2
        image = image.crop((0, top, ori_w, bottom))
    image = image.resize((target_width, target_height), resample=Image.LANCZOS)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])(image)
    return image.to(device)


if __name__ == "__main__":
    parser = parse_args()
    opt, unknown = parser.parse_known_args()

    set_lowvram_mode(opt.low_vram)
    version_dict = VERSION2SPECS[opt.version]
    model = init_model(version_dict)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    report_interval = 100
    if opt.fid or opt.fvd or opt.fvd_cuda:
        metric_save_path = os.path.join(opt.save, "metrics.txt")
        f = open(metric_save_path, mode='w')

    if opt.fid:
        fid = FrechetInceptionDistance()
    if opt.fid_dist:
        fid = FlexibleFrechetInceptionDistance()
        real_fid_feats_list = list()
        fake_fid_feats_list = list()

    if opt.fvd or opt.fvd_cuda or opt.fvd_dist:
        print('load i3d ...')
        device = torch.device('cpu') if opt.fvd else torch.device('cuda')
        i3d = load_fvd_model(device)
        real_embeddings_list = list()
        fake_embeddings_list = list()
        assert opt.fvd_version in ['v1', 'v2']
        vista_official_fvd = opt.fvd_version == 'v2'

    # depth metrics
    if opt.depth_eval:
        metric_funcs = [getattr(metric, _met) for _met in DEPTH_EVAL_METRICS]

        metric_tracker = DepthMetricTracker(*[m.__name__ for m in metric_funcs])
        metric_tracker.reset()  
    if opt.save_depth:
        depth_pred_list = list()
        depth_target_list = list()

    sample_index = opt.sample_index
    max_sample_index = 1000000000 if opt.num_samples is None else sample_index + opt.num_samples
    while sample_index >= 0:
        seed_everything(opt.seed)

        frame_list, sample_index, dataset_length, action_dict = get_sample(sample_index,
                                                                           opt.dataset,
                                                                           opt.n_frames,
                                                                           opt.action)

        img_seq = list()
        for each_path in frame_list:
            img = load_img(each_path, opt.height, opt.width)
            img_seq.append(img)
        images = torch.stack(img_seq)

        value_dict = init_embedder_options(unique_keys)
        cond_img = img_seq[0][None]
        value_dict["cond_frames_without_noise"] = cond_img
        value_dict["cond_aug"] = opt.cond_aug
        value_dict["cond_frames"] = cond_img + opt.cond_aug * torch.randn_like(cond_img)
        if action_dict is not None:
            for key, value in action_dict.items():
                value_dict[key] = value

        if opt.n_rounds > 1:
            guider = "TrianglePredictionGuider"
        else:
            guider = "VanillaCFG"
        sampler = init_sampling(guider=guider, steps=opt.n_steps, cfg_scale=opt.cfg_scale, num_frames=opt.n_frames)

        uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]

        out = do_sample(
            images,
            model,
            sampler,
            value_dict,
            num_rounds=opt.n_rounds,
            num_frames=opt.n_frames,
            force_uc_zero_embeddings=uc_keys,
            initial_cond_indices=[index for index in range(opt.n_conds)],
            perception_target_vis=False,
            perception_target_using_sample=True
        )

        if isinstance(out, (tuple, list)):
            samples, samples_z, inputs, depth, depth_target, depth_rec = out
            virtual_path = os.path.join(opt.save, "virtual")
            real_path = os.path.join(opt.save, "real")

            if opt.fid or opt.fid_dist:  # calculate FID score here
                inputs_img = 255.0 * (inputs + 1.0) / 2.0
                samples_img = 255.0 * samples

                if opt.fid_dist:  # save all images to disk for each thread
                    real_feat = fid.extract_feats(inputs_img.cpu().type(torch.uint8))
                    fake_feat = fid.extract_feats(samples_img.cpu().type(torch.uint8))
                    real_fid_feats_list.append(real_feat)
                    fake_fid_feats_list.append(fake_feat)
                else:
                    fid.update(inputs_img.cpu().type(torch.uint8), real=True)
                    fid.update(samples_img.cpu().type(torch.uint8), real=False)

                    if sample_index > 0 and sample_index % report_interval == 0:
                        cur_fid = fid.compute()
                        print(f"First {sample_index} samples FID: {cur_fid}", file=f)

            if opt.fvd or opt.fvd_cuda or opt.fvd_dist:  # calculate FVD score here
                input_videos = 255.0 * (inputs + 1.0) / 2.0   # num_frames x 3 x H x W
                samples_videos = 255.0 * samples   # num_frames x 3 x H x W
                input_videos = input_videos.permute(0, 2, 3, 1).unsqueeze(0)  # 1 x num_frames x H x W x 3
                samples_videos = samples_videos.permute(0, 2, 3, 1).unsqueeze(0)
                if not opt.fvd_cuda:
                    input_videos = input_videos.cpu()
                    samples_videos = samples_videos.cpu()

                real_embeddings = get_fvd_logits(input_videos, i3d, device=device, batch_size=1, pre_resize=vista_official_fvd)
                fake_embeddings = get_fvd_logits(samples_videos, i3d, device=device, batch_size=1, pre_resize=vista_official_fvd)
                real_embeddings_list.append(real_embeddings)
                fake_embeddings_list.append(fake_embeddings)

                if not opt.fvd_dist:
                    if sample_index > 0 and sample_index % report_interval == 0:
                        fake = torch.cat(fake_embeddings_list, dim=0)
                        real = torch.cat(real_embeddings_list, dim=0)
                        cur_fvd = frechet_distance(fake, real)
                        cur_fvd = cur_fvd.cpu().numpy()
                        print(f"First {sample_index} samples FVD: {cur_fvd}")

            perform_save_locally(virtual_path, samples, "videos", opt.dataset, sample_index)
            perform_save_locally(virtual_path, samples, "grids", opt.dataset, sample_index)
            perform_save_locally(virtual_path, samples, "images", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "videos", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "grids", opt.dataset, sample_index)
            perform_save_locally(real_path, inputs, "images", opt.dataset, sample_index)
            if depth is not None:
                depth_path = os.path.join(opt.save, "virtural_depth")
                depth_vis = get_depth_vis(img=None, depth=depth, to_numpy=False)
                depth_vis = torch.stack(depth_vis)
                depth_vis = (depth_vis + 1.0) / 2.0
                perform_save_locally(depth_path, depth_vis, "videos", opt.dataset, sample_index)
                perform_save_locally(depth_path, depth_vis, "grids", opt.dataset, sample_index)
                perform_save_locally(depth_path, depth_vis, "images", opt.dataset, sample_index)
            if depth_target is not None:
                depth_target_path = os.path.join(opt.save, "virtural_depth_target")
                depth_target_vis = get_depth_vis(img=None, depth=depth_target, to_numpy=False)
                depth_target_vis = torch.stack(depth_target_vis)
                depth_target_vis = (depth_target_vis + 1.0) / 2.0  # suitable for perform_save_locally func
                perform_save_locally(depth_target_path, depth_target_vis, "videos", opt.dataset, sample_index)
                perform_save_locally(depth_target_path, depth_target_vis, "grids", opt.dataset, sample_index)
                perform_save_locally(depth_target_path, depth_target_vis, "images", opt.dataset, sample_index)
            if depth_rec is not None:
                depth_rec_path = os.path.join(opt.save, "virtural_depth_rec_from_AE")
                perform_save_locally(depth_rec_path, depth_rec, "videos", opt.dataset, sample_index)
                perform_save_locally(depth_rec_path, depth_rec, "grids", opt.dataset, sample_index)
                perform_save_locally(depth_rec_path, depth_rec, "images", opt.dataset, sample_index)

            if opt.depth_eval:
                assert depth is not None and depth_target is not None
                eval_depth(depth.float(), ((depth_target + 1.0) / 2.0).float(), metric_funcs, metric_tracker)
            if opt.save_depth:
                assert depth is not None and depth_target is not None
                depth_pred_list.append(depth.float())
                depth_target_list.append(((depth_target + 1.0) / 2.0).float())
        else:
            raise TypeError

        if opt.rand_gen:
            sample_index += random.randint(1, dataset_length - 1)
        else:
            sample_index += 1
            if min(dataset_length, max_sample_index) <= sample_index:
                sample_index = -1

    # calculate FID
    if opt.fid or opt.fid_dist:
        if not opt.fid_dist:
            print(f'Total sample num: {min(max_sample_index, dataset_length) - opt.sample_index}, FID: {fid.compute()}', file=f)
        else:   # save to disk
            real_fid_feats = torch.cat(real_fid_feats_list, dim=0)
            fake_fid_feats = torch.cat(fake_fid_feats_list, dim=0)
            
            fid_save_dir = os.path.join(opt.save, 'feats_saved_for_metric')
            os.makedirs(fid_save_dir, exist_ok=True)
            cur_device = os.environ['CUDA_VISIBLE_DEVICES']
            fid_save_path = os.path.join(fid_save_dir, f'cuda_{cur_device}_fid_feats.pt')
            torch.save(dict(real_feats=real_fid_feats, fake_feats=fake_fid_feats), fid_save_path)

    # calculate FVD
    if opt.fvd or opt.fvd_dist:
        fake = torch.cat(fake_embeddings_list, dim=0)
        real = torch.cat(real_embeddings_list, dim=0)
        if not opt.fvd_dist:
            cur_fvd = frechet_distance(fake, real)
            cur_fvd = cur_fvd.cpu().numpy()
            print(f'Total sample num: {len(fake_embeddings_list)}, FVD: {cur_fvd}', file=f)
        else:
            fvd_save_dir = os.path.join(opt.save, 'feats_saved_for_metric')
            os.makedirs(fvd_save_dir, exist_ok=True)
            cur_device = os.environ['CUDA_VISIBLE_DEVICES']
            fvd_save_path = os.path.join(fvd_save_dir, f'cuda_{cur_device}_fvd_feats.pt')
            torch.save(dict(fake_feats=fake, real_feats=real), fvd_save_path)

    if opt.fid or opt.fvd or opt.fvd_cuda:
        f.close()

    # depth evaluation
    if opt.depth_eval:
        eval_text = f"Depth evaluation metrics:\n"

        eval_text += tabulate(
            [metric_tracker.result().keys(), metric_tracker.result().values()]
        )
        print(eval_text)

        metrics_filename = "depth_eval_metrics.txt"

        _save_to = os.path.join(opt.save, metrics_filename)
        with open(_save_to, "w+") as f:
            f.write(eval_text)
            logging.info(f"Depth evaluation metrics saved to {_save_to}")

    if opt.save_depth:
        depth_save_dir = os.path.join(opt.save, 'depth_saved_for_metric')
        os.makedirs(depth_save_dir, exist_ok=True)
        cur_device = os.environ['CUDA_VISIBLE_DEVICES']
        depth_save_path = os.path.join(depth_save_dir, f'cuda_{cur_device}_depth.pt')
        torch.save(dict(pred=depth_pred_list, target=depth_target_list), depth_save_path)
