import os
import argparse
import random
import time
import json
import gzip
from tqdm import tqdm

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from utils import normalize, get_transform


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_level_metrics(gt, pr, metric_name):
    if metric_name == 'auroc':
        return roc_auc_score(gt, pr)
    elif metric_name == 'ap':
        return average_precision_score(gt, pr)
    elif metric_name == 'f1_score':
        return f1_score(gt, pr)


def pixel_level_metrics(gt, pr, metric_name):
    if metric_name == 'auroc':
        return roc_auc_score(gt.ravel(), pr.ravel())
    elif metric_name == 'ap':
        return average_precision_score(gt.ravel(), pr.ravel())
    elif metric_name == 'f1_score':
        thresholds = np.linspace(0, 1, 100)
        f1s = []
        for t in thresholds:
            y_pred = (pr.ravel() >= t).astype(int)
            f1s.append(f1_score(gt.ravel(), y_pred, zero_division=0))
        idx_f1 = np.argmax(f1s)
        return f1s[idx_f1], thresholds[idx_f1]
    

def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, alpha):
    for idx, path in enumerate(pathes):
        specie_name = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask, alpha)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = os.path.join(save_path, 'heatmaps', cls_name, specie_name)
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(os.path.join(save_vis, filename), vis)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def test(args):
    start_time_preprocess = time.perf_counter()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    AnomalyCLIP_parameters = {'Prompt_length': args.n_ctx, 'learnabel_text_embedding_depth': args.depth, 'learnabel_text_embedding_length': args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load('ViT-L/14@336px', device=device, design_details=AnomalyCLIP_parameters, download_root=args.clip_model_path)
    model.eval() # freeze clip-model

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.dataset_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset_name, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list
    
    prompt_learner = AnomalyCLIP_PromptLearner(model.to(device), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint['prompt_learner'])
    prompt_learner.to(device)
    # prompt_learner.eval() # for CoCoOp's principle
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    """
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None) # forward method of AnomalyCLIP_PromptLearner
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    """
    model.to(device)

    end_time_preprocess = time.perf_counter()

    # create folder structure
    # os.makedirs(f'{args.save_path}/features', exist_ok=True)
    # np.save(f'{args.save_path}/features/text_features.npy', text_features.detach().cpu().numpy())

    # for saving test results
    test_info = dict(times={}, test_results={})
    results = {obj: {'gt_scores': [], 'pr_scores': [], 'gt_masks': [], 'heatmaps': []} for obj in obj_list}
    metric_results = dict(metric_results={obj: {} for obj in obj_list})

    start_time_total_test = time.perf_counter()

    for idx_example, items in enumerate(tqdm(test_dataloader)):
        start_time_per_img = time.perf_counter()

        image = items['img'].to(device)
    
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=args.dpam_layer)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # new idea (CoCoOp's principle)
            prompts, tokenized_prompts, compound_prompts_text = prompt_learner(image_features, patch_features, mode=None, cls_id=None) # patch_features # forward method of AnomalyCLIP_PromptLearner
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            text_probs = text_probs[:, 0, 1]

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            anomaly_map = torch.stack(anomaly_map_list)
            anomaly_map = anomaly_map.sum(dim=0)
            anomaly_map = torch.stack([torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()], dim=0) # auskommentieren!

            anomaly_map = np.squeeze(anomaly_map.detach().cpu().numpy())
            pr_sp = float(text_probs.detach().cpu())

        end_time_per_img = time.perf_counter()
        time_per_img = end_time_per_img - start_time_per_img

        ##################################################
        data = test_data.data_all[idx_example]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], data['specie_name'], data['anomaly']

        ##################################################
        # save stuff for metrics computation
        heatmap = normalize(anomaly_map)
        results[cls_name]['heatmaps'].append(heatmap) # .detach().cpu().numpy())
        results[cls_name]['gt_masks'].append(gt_mask.detach().cpu().numpy())
        results[cls_name]['gt_scores'].append(anomaly)
        results[cls_name]['pr_scores'].append(pr_sp)
        
        ##################################################
        # save results for i-th example
        os.makedirs(f'{args.save_path}/features/{cls_name}/{specie_name}/text_features', exist_ok=True) # for CoCoOp's principle
        os.makedirs(f'{args.save_path}/features/{cls_name}/{specie_name}/image_features', exist_ok=True)
        os.makedirs(f'{args.save_path}/features/{cls_name}/{specie_name}/patch_features', exist_ok=True)
        # np.save(f'{args.save_path}/features/{cls_name}/{specie_name}/text_features/{img_path[-7:-4]}.npy', text_features.detach().cpu().numpy()) # for CoCoOp's principle
        # np.save(f'{args.save_path}/features/{cls_name}/{specie_name}/image_features/{img_path[-7:-4]}.npy', image_features.detach().cpu().numpy())
        # np.save(f'{args.save_path}/features/{cls_name}/{specie_name}/patch_features/{img_path[-7:-4]}.npy', [pf.detach().cpu().numpy() for pf in patch_features], allow_pickle=True)
        """
        f1 = gzip.GzipFile(f'{args.save_path}/features/{cls_name}/{specie_name}/text_features/{img_path[-7:-4]}.npy.gz', 'w')
        np.save(file=f1, arr=text_features.detach().cpu().numpy()) # for CoCoOp's principle
        f1.close()

        f2 = gzip.GzipFile(f'{args.save_path}/features/{cls_name}/{specie_name}/image_features/{img_path[-7:-4]}.npy.gz', 'w')
        np.save(file=f2, arr=image_features.detach().cpu().numpy())
        f2.close()

        f3 = gzip.GzipFile(f'{args.save_path}/features/{cls_name}/{specie_name}/patch_features/{img_path[-7:-4]}.npy.gz', 'w')
        np.save(file=f3, arr=[pf.detach().cpu().numpy() for pf in patch_features], allow_pickle=True)
        f3.close()
        """

        visualizer(items['img_path'], anomaly_map, args.image_size, args.save_path, cls_name, args.alpha)
                                                         
        info_img = dict(
            img_path=f'{cls_name}/test/{specie_name}/{img_path[-7:]}',
            mask_path=f'{cls_name}/ground_truth/{specie_name}/{mask_path[-12:]}' if specie_name not in ['good'] else '',
            hm_path=f'heatmaps/{cls_name}/{specie_name}/{img_path[-7:-4]}.png',
            cls_name=cls_name,
            specie_name=specie_name,
            anomaly=anomaly,
            pr_sp=pr_sp,
            time=time_per_img,
        )

        if cls_name not in test_info['test_results'].keys():
            test_info['test_results'][cls_name] = []
        
        test_info['test_results'][cls_name].append(info_img)
        
    end_time_total_test = time.perf_counter()

    test_info['times']['time_preprocess'] = end_time_preprocess - start_time_preprocess
    test_info['times']['time_total_test'] = end_time_total_test - start_time_total_test  

    ##################################################
    # save test results for all images
    with open(f'{args.save_path}/test_results.json', 'w') as f:
        f.write(json.dumps(test_info, indent=4) + '\n')

    ##################################################
    # compute metrics
    os.makedirs(f'{args.save_path}/metrics', exist_ok=True)

    for obj in obj_list:
        gt_scores = np.asarray(results[obj]['gt_scores'])
        pr_scores = np.asarray(results[obj]['pr_scores'])
        gt_masks = np.asarray(results[obj]['gt_masks'])
        heatmaps = np.asarray(results[obj]['heatmaps'])

        metric_results['metric_results'][obj]['image_auroc'] = image_level_metrics(gt_scores, pr_scores, 'auroc')
        metric_results['metric_results'][obj]['image_ap'] = image_level_metrics(gt_scores, pr_scores, 'ap')
        metric_results['metric_results'][obj]['pixel_auroc'] = pixel_level_metrics(gt_masks, heatmaps, 'auroc')
        metric_results['metric_results'][obj]['pixel_ap'] = pixel_level_metrics(gt_masks, heatmaps, 'ap')
        # metric_results['metric_results'][obj]['pixel_max_f1_score'], metric_results['metric_results'][obj]['threshold'] = pixel_level_metrics(gt_masks, heatmaps, 'f1_score')

    with open(f'{args.save_path}/metrics/metric_results.json', 'w') as f:
        f.write(json.dumps(metric_results, indent=4) + '\n')


if __name__ == '__main__':
    project_dir = os.getcwd() + '/AnomalyCLIP'
    print(project_dir)

    parser = argparse.ArgumentParser('Test AnomalyCLIP', add_help=True)

    parser.add_argument('--dataset_path', type=str, default=f'{project_dir}/data/dataset_masterproject', help='path to test dataset')
    parser.add_argument('--checkpoint_path', type=str, default=f'{project_dir}/checkpoints/train_acbig_adapter_run2/epoch_15.pth', help='path to prompt-learner checkpoint')
    parser.add_argument('--save_path', type=str, default=f'{project_dir}/results/test_acbig_adapter_run2', help='path to save results')
    parser.add_argument('--clip_model_path', type=str, default=f'{project_dir}/clip_model', help='path to clip model')
    
    parser.add_argument('--dataset_name', type=str, default='masterproject', help='test dataset name')
    parser.add_argument('--features_list', type=int, nargs='+', default=[6, 12, 18, 24], help='features used')
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--dpam_layer', type=int, default=20, help='dpam layer amount')
    parser.add_argument('--depth', type=int, default=9, help='image size')
    parser.add_argument('--n_ctx', type=int, default=12, help='zero shot')
    parser.add_argument('--t_n_ctx', type=int, default=4, help='zero shot')
    parser.add_argument('--feature_map_layer', type=int,  nargs='+', default=[0, 1, 2, 3], help='zero shot')
    parser.add_argument('--metrics', type=str, default='image-pixel-level')
    parser.add_argument('--seed', type=int, default=111, help='random seed')
    parser.add_argument('--sigma', type=int, default=4, help='std of gaussian filter kernel')
    parser.add_argument('--alpha', type=int, default=0.5, help='alpha for applying scoremap')

    args = parser.parse_args()
    print(args)
    setup_seed(args.seed)
    test(args)