"""Compute metrics."""

import os
import argparse
import json
from tqdm import tqdm
from PIL import Image
import sys

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from Programme.Visualization.VisualizeResults import AnomalyEvaluator


def normalize(heatmap):
    return (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())


def image_level_metrics(gt, pr):
    return roc_auc_score(gt, pr), average_precision_score(gt, pr)#, f1_score(gt, pr)


def pixel_level_metrics(gt, pr):
    return roc_auc_score(gt.ravel(), pr.ravel())#, f1_score(gt.ravel(), pr.ravel())

def pixel_level_f1(gt, pr):
    return f1_score(gt.ravel(), pr.ravel())


def compute_metrics(args):
    test_results_info = json.load(open(f'{os.path.join(args.test_results_path)}/{args.category}_test_results.json', 'r'))

    # times = test_results_info['times']
    test_results = test_results_info['test_results']

    metric_results = dict(metric_results={})

    for obj in tqdm(test_results.keys()):
        info = test_results[obj]

        if obj not in metric_results['metric_results'].keys():
            metric_results['metric_results'][obj] = {}

        gt_scores = []
        pr_scores = []
        heatmaps = []
        masks = []

        for idx_example, items in enumerate(info):
            try: 
                heatmap = cv2.cvtColor(cv2.imread(f"{args.test_results_path}/{items['hm_path']}"), cv2.COLOR_BGR2RGB) # Image.open(f'{args.test_results_path}/{items['hm_path'][19:]}')
                heatmap = normalize(heatmap.mean(axis=2))

                mask_path = items['mask_path']
                if mask_path:
                    mask = cv2.cvtColor(cv2.imread(f'{args.dataset_path}/{mask_path}'), cv2.COLOR_BGR2GRAY) # Image.open(f'{args.dataset_path}/{mask_path[28:]}')
                    mask = cv2.resize(mask, (heatmap.shape[0], heatmap.shape[1]))
                    mask = (mask == 255).astype(np.float32)
                else:
                    mask = np.zeros_like(heatmap)

                heatmaps.append(heatmap)
                masks.append(mask)

                gt_scores.append(items['anomaly'])
                pr_scores.append(items['pr_sp'])
            except Exception as e:
                print("could not open Heatmap:")
                print(f"{args.test_results_path}/{items['hm_path']}")
                print(e)


        heatmaps = np.asarray(heatmaps)
        masks = np.asarray(masks)
        metric_results['metric_results'][obj]['pixel_level_auroc'] = pixel_level_metrics(gt=masks, pr=heatmaps)

        gt_scores = np.asarray(gt_scores)
        pr_scores = np.asarray(pr_scores)
        metric_results['metric_results'][obj]['image_level_auroc'] = image_level_metrics(gt=gt_scores, pr=pr_scores)

    save_path = f'{args.save_path}/{args.category}_metric_results.json'
    with open(save_path, 'w') as f:
        f.write(json.dumps(metric_results, indent=4) + '\n')
    print(f"saved test results to {save_path}")

def pixelF1(args):
    test_results_info = json.load(open(f'{os.path.join(args.test_results_path)}/{args.category}_test_results.json', 'r'))

    # times = test_results_info['times']
    test_results = test_results_info['test_results']

    for obj in tqdm(test_results.keys()):
        info = test_results[obj]
        heatmaps = []
        masks = []
        
        for idx_example, items in enumerate(info):
            try: 
                heatmap = cv2.cvtColor(cv2.imread(f"{args.test_results_path}/{items['hm_path']}"), cv2.COLOR_BGR2RGB) # Image.open(f'{args.test_results_path}/{items['hm_path'][19:]}')
                heatmap = normalize(heatmap.mean(axis=2))

                mask_path = items['mask_path']
                if mask_path:
                    mask = cv2.cvtColor(cv2.imread(f'{args.dataset_path}/{mask_path}'), cv2.COLOR_BGR2GRAY) # Image.open(f'{args.dataset_path}/{mask_path[28:]}')
                    mask = cv2.resize(mask, (heatmap.shape[0], heatmap.shape[1]))
                    mask = (mask == 255).astype(np.float32)
                else:
                    mask = np.zeros_like(heatmap)

                heatmaps.append(heatmap)
                masks.append(mask)

            except Exception as e:
                print("could not open Heatmap:")
                print(f"{args.test_results_path}/{items['hm_path']}")
                print(e)


        heatmaps = np.asarray(heatmaps)
        threshold = 0.5
        binaryHeatmaps = (heatmaps > threshold).astype(int)

        masks = np.asarray(masks)
        metric_results = dict()
        print(heatmaps.shape, masks.shape)
        metric_results['pixel_level_f1'] = pixel_level_f1(gt=masks, pr=binaryHeatmaps)

        evaluator = AnomalyEvaluator(masks.ravel(), heatmaps.ravel())
        print(evaluator.get_best_threshold())

    save_path = f'{args.save_path}/{args.category}_metric_results.json'

    metricsJSON = save_path
    with open(metricsJSON) as f_out:
        metrics = json.load(f_out)
    metrics['metric_results'][args.category].update(metric_results)
    with open(metricsJSON, "w") as f_out:
        json.dump(metrics, f_out, indent=4)

def define_parser():
    project_path = os.getcwd()
    parser = argparse.ArgumentParser('Compute Metrics', add_help=True)
    parser.add_argument('--dataset_path', type=str, default=f'{project_path}/AnomalyCLIP/data/dataset_masterproject', help='path to test dataset')
    parser.add_argument('--test_results_path', type=str, default=f'{project_path}/AnomalyCLIP/results/test_vlm', help='path to test results')
    parser.add_argument('--save_path', type=str, default=f'{project_path}/Evaluation/results', help='path to save results')
    parser.add_argument('--category', type=str, required=True, help='category to test model on')

    return parser.parse_args()

if __name__ == '__main__':
    project_path = os.getcwd()

    print(f'Project path: {project_path}')

    # args = define_parser()
    # print(args)

    sys.argv = [
        'train_and_test.py',  # Name des Skripts
        '--dataset_path', '/Volumes/Extreme SSD/Master/Projekt/datasets/MVTecAD',
        '--test_results_path', '/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_efficientAD2',
        '--save_path', '/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_efficientAD2',
        '--category', 'bottle',
    ]
    args = define_parser()
    print(args)
    #compute_metrics(args)
    pixelF1(args)