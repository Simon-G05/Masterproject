import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform
from dataset import Dataset


def helper(args):
    os.makedirs(args.save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    AnomalyCLIP_parameters = {"Prompt_length": args.n_ctx, "learnabel_text_embedding_depth": args.depth, "learnabel_text_embedding_length": args.t_n_ctx}

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters, download_root=args.clip_model_path)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(root=args.data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to(device), AnomalyCLIP_parameters) # "cpu"
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id = None)
    
    print(f'prompts.shape: {prompts.shape} | tokenized_prompts.shape: {tokenized_prompts.shape}') #  | compound_prompts_text.shape: {compound_prompts_text.shape}')

    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)

    model.to(device)
    """
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name'][0]
        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer = 20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ text_features.permute(0, 2, 1)
            text_probs = (text_probs/0.07).softmax(-1)
            # text_probs = text_probs[:, 0, 1]
    """
    """
        np.save(f'{args.save_path}image_features_{cls_name}_{idx}.npy', image_features.detach().cpu().numpy())
        np.save(f'{args.save_path}patch_features_{cls_name}_{idx}.npy', [pf.detach().cpu().numpy() for pf in patch_features], allow_pickle=True)
        np.save(f'{args.save_path}text_probs_{cls_name}_{idx}.npy', text_probs.detach().cpu().numpy())
    """
    np.save(f'{args.save_path}prompts.npy',  prompts.detach().cpu().numpy())
    np.save(f'{args.save_path}tokenized_prompts.npy', tokenized_prompts.detach().cpu().numpy())
    np.save(f'{args.save_path}compound_prompts_text.npy', [p.detach().cpu().numpy() for p in compound_prompts_text], allow_pickle=True)
    np.save(f'{args.save_path}text_features.npy', text_features.detach().cpu().numpy())
    

    # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # print(sum(p.numel() for p in prompt_learner.parameters() if p.requires_grad))

if __name__ == '__main__':
    project_path = os.getcwd()

    parser = argparse.ArgumentParser("Helper", add_help=True)
    
    parser.add_argument("--data_path", type=str, default="./data/dataset_few_shot/4", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default=f'{project_path}/helper_results/zero_shot/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/train_acbig_run1/epoch_27.pth', help='path to checkpoint') # few_shot/4/epoch_27.pth
    parser.add_argument("--clip_model_path", type=str, default='./clip_model', help='path to clip model')
    
    parser.add_argument("--dataset", type=str, default='masterproject', help='name of dataset')
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="zero shot")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="zero shot")
    
    args = parser.parse_args()
    print(args)
    helper(args)