import argparse
import torch
import FSPNet_model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
import cv2
from tqdm import tqdm
import time  # For FPS measurement


def evaluator(gt_pth_lst, pred_pth_lst, metrics):
    module_map_name = {
        "Smeasure": "Smeasure", "wFmeasure": "WeightedFmeasure", "MAE": "MAE",
        "adpEm": "Emeasure", "meanEm": "Emeasure", "maxEm": "Emeasure",
        "adpFm": "Fmeasure", "meanFm": "Fmeasure", "maxFm": "Fmeasure",
        "meanSen": "Medical", "maxSen": "Medical", "meanSpe": "Medical", "maxSpe": "Medical",
        "meanDice": "Medical", "maxDice": "Medical", "meanIoU": "Medical", "maxIoU": "Medical"
    }
    res, metric_module = {}, {}
    metric_module_list = [module_map_name[metric] for metric in metrics]
    metric_module_list = list(set(metric_module_list))

    # Define measures
    for metric_module_name in metric_module_list:
        metric_module[metric_module_name] = getattr(
            __import__("metrics", fromlist=[metric_module_name]),
            metric_module_name
        )(length=len(gt_pth_lst))

    assert len(gt_pth_lst) == len(pred_pth_lst)

    # Evaluator
    for idx in tqdm(range(len(gt_pth_lst))):
        gt_pth = gt_pth_lst[idx]
        pred_pth = pred_pth_lst[idx]
        assert os.path.isfile(gt_pth) and os.path.isfile(pred_pth)

        pred_ary = cv2.imread(pred_pth, cv2.IMREAD_GRAYSCALE)
        gt_ary = cv2.imread(gt_pth, cv2.IMREAD_GRAYSCALE)

        # Ensure shape match
        if not gt_ary.shape == pred_ary.shape:
            pred_ary = cv2.resize(pred_ary, (gt_ary.shape[1], gt_ary.shape[0]))

        for module in metric_module.values():
            module.step(pred=pred_ary, gt=gt_ary, idx=idx)

    for metric in metrics:
        module = metric_module[module_map_name[metric]]
        results = module.get_results()[metric]
        if isinstance(results, np.ndarray):
            res[metric] = np.mean(results)
        else:
            res[metric] = results

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run FSPNet model on datasets.")
    parser.add_argument("--ckpt_path", required=True, help="Path to checkpoint.")
    parser.add_argument("--result_save_root", required=True, help="Path to save the results.")
    args = parser.parse_args()

    batch_size = 1
    net = FSPNet_model.Model(None, img_size=384).cuda()

    # Load checkpoint
    ckpt_path = args.ckpt_path
    pretrained_dict = torch.load(ckpt_path)
    net_dict = net.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    net.eval()

    # Test datasets
    Dirs = [
        "/mnt/scratch/scrmat/PolypData/kvasir_ses/",
        "/mnt/scratch/scrmat/PolypData/Etis_Larib/",
        "/mnt/scratch/scrmat/PolypData/CVC-ColonDB/",
        "/mnt/scratch/scrmat/PolypData/C6/"
    ]

    metrics = ["meanIoU", "meanDice", "Smeasure", "wFmeasure", "MAE"]
    dataset_results = {}

    for dataset_dir in Dirs:
        dataset_name = dataset_dir.split("/")[-2]
        print(f"Processing dataset: {dataset_name}")

        save_path = os.path.join(args.result_save_root, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        Dataset = dataset.TestDataset(dataset_dir, 384)
        Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size * 2)

        gt_pth_lst = []
        pred_pth_lst = []

        total_time = 0.0
        total_frames = 0

        for data in tqdm(Dataloader):
            img, label = data['img'].cuda(), data['label'].cuda()
            name = data['name'][0].split("/")[-1]

            start_time = time.time()

            with torch.no_grad():
                mask_out, edge_out = net(img)
                out = mask_out[3]

            end_time = time.time()
            total_time += (end_time - start_time)
            total_frames += img.size(0)  # batch_size=1 but kept general

            B, C, H, W = label.size()
            o = F.interpolate(out, (H, W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0, 0]
            o = (o - o.min()) / (o.max() - o.min() + 1e-8)
            o_bin = (o > 0.5).astype(np.uint8)
            o_bin = (o_bin * 255).astype(np.uint8)

            pred_pth = os.path.join(save_path, name)
            imwrite(pred_pth, o_bin)

            gt_pth = os.path.join(dataset_dir, "masks", name)
            gt_pth_lst.append(gt_pth)
            pred_pth_lst.append(pred_pth)

        # Calculate FPS
        fps = total_frames / total_time if total_time > 0 else 0.0

        # Evaluate metrics
        results = evaluator(gt_pth_lst, pred_pth_lst, metrics)
        results['FPS'] = fps  # Append FPS to results
        dataset_results[dataset_name] = results

        print(f"Results for dataset {dataset_name}:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")

    print("\nSummary of Results:")
    for dataset_name, results in dataset_results.items():
        print(f"Dataset: {dataset_name}")
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
