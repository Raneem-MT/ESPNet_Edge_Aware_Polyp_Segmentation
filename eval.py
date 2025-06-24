
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
from py_sod_metrics.sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


if __name__ == '__main__':
    batch_size = 1
    net = FSPNet_model.Model(None, img_size=384).cuda()

    # Load checkpoint
    ckpt_path = "/resstore/b0211/Users/scrmat/pgen_fsp/checkpoints/model_118_loss_0.00516.pth"
    pretrained_dict = torch.load(ckpt_path)
    net_dict = net.state_dict()
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in net_dict}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    net.eval()

    # Directories containing test data
    Dirs = [
        "/resstore/b0211/Data/polypData/sunhard_seen/",
        "/resstore/b0211/Data/polypData/leeds_wli/",
    ]

    # Directory to save results
    result_save_root = "/resstore/b0211/Users/scrmat/pgen_fsp/Results/"

    # Metrics to evaluate
    metrics = ["meanIoU", "meanDice", "Smeasure", "wFmeasure", "MAE"]

    # Dictionary to store results for each dataset
    dataset_results = {}

    for dataset_dir in Dirs:
        dataset_name = dataset_dir.split("/")[-2]
        print(f"Processing dataset: {dataset_name}")

        # Create directories for saving results if they don't exist
        save_path = os.path.join(result_save_root, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        # Load dataset
        Dataset = dataset.TestDataset(dataset_dir, 384)
        Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size * 2)

        # Lists to store paths of ground truth and predictions for this dataset
        gt_pth_lst = []
        pred_pth_lst = []

        # Generate predictions for the dataset
        for data in tqdm(Dataloader):
            img, label = data['img'].cuda(), data['label'].cuda()
            name = data['name'][0].split("/")[-1]

            with torch.no_grad():
                out = net(img)[3]  # Assuming the 4th output is the segmentation mask

            B, C, H, W = label.size()
            o = F.interpolate(out, (H, W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0, 0]
            o = (o - o.min()) / (o.max() - o.min() + 1e-8)
            o_bin = (o > 0.5).astype(np.uint8)  
            o_bin = (o_bin * 255).astype(np.uint8)

            # Save prediction
            pred_pth = os.path.join(save_path, name)
            imwrite(pred_pth, o_bin)

            # Save ground truth path
            gt_pth = os.path.join(dataset_dir, "masks", name)
            gt_pth_lst.append(gt_pth)
            pred_pth_lst.append(pred_pth)

        
        mask_root = os.path.join(dataset_dir, "masks")  
        mask_name_list = sorted(os.listdir(mask_root))
        FM = Fmeasure()
        WFM = WeightedFmeasure()
        SM = Smeasure()
        EM = Emeasure()
        M = MAE()



        for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
            mask_path = os.path.join(mask_root, mask_name)  
            pred_path = os.path.join(save_path, mask_name) 
          


            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            FM.step(pred=pred, gt=mask)
            WFM.step(pred=pred, gt=mask)
            SM.step(pred=pred, gt=mask)
            EM.step(pred=pred, gt=mask)
            M.step(pred=pred, gt=mask)


    
        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = M.get_results()["mae"]

        results = {
            "Smeasure": sm,
            "wFmeasure": wfm,
            "MAE": mae,
        }

        print(f"Results: {results}")
