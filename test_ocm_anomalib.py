import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections.abc import Mapping
import os
import argparse
import random
import time
import json
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from anomalib.callbacks import GraphLogger
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.models import get_model  # Anomalib 2.0.0 API
from anomalib.data import MVTecAD, get_datamodule  # Neu in 2.0.0
from anomalib.engine import Engine   # Engine für Training und Testen
from omegaconf import DictConfig, OmegaConf

from utils import normalize, get_transform, saveHeatmap
from pytorch_lightning.callbacks import Callback


class LossLoggerCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # outputs muss den Loss enthalten, typischerweise outputs['loss']
        loss = outputs['loss'].item() if isinstance(outputs, dict) and 'loss' in outputs else None
        if loss is not None:
            # Log den Loss an TensorBoard
            trainer.logger.experiment.add_scalar("train/loss", loss, trainer.global_step)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_time_preprocess = time.perf_counter()

    # Lade das Modell über den get_model Mechanismus (Anomalib 2.0.0)
    try:
        config = OmegaConf.load(args.model_config)
        model = get_model(config.model)
    except:
        model = get_model(model=args.model_name) 
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # Lade das Datamodul
    # Lade Datamodul mit Batchgröße 1
    data_config = DictConfig({
        "data": {
            "class_path": "MVTecAD",
            "init_args": {
                "root": args.dataset_path,
                "category": args.category,
            },
            "transform": {
                "image_size": [args.image_size, args.image_size],
            },
        }
    })

    datamodule = get_datamodule(config=data_config)
    datamodule.eval_batch_size = 1       
    datamodule.setup(stage="test")  # <-- Das initialisiert test_data
    test_dataloader = datamodule.test_dataloader()

    end_time_preprocess = time.perf_counter()

    test_info = dict(times={}, test_results={})

    start_time = time.perf_counter()

    for idx_example, items in enumerate(tqdm(test_dataloader, desc="Testing")):
        image = items.image.to(device)

        with torch.inference_mode():
            prediction = model(image)
            anomaly_map = prediction.anomaly_map.cpu().numpy().squeeze()
            anomaly_map = np.squeeze(anomaly_map)

            pred_label = prediction.pred_label
            pr_sp = float(prediction.pred_score.item())  # Klassifikationsscore als float


        # Bilddaten extrahieren
        data = datamodule.test_data[idx_example]
        img_path, mask_path, cls_name, specie_name = data.image_path, data.mask_path, datamodule.category, data.image_path.split("/")[-2]
        anomaly = 0 if specie_name in ['Good', 'good'] else 1

        save_path = f'{args.save_path}/heatmaps/{cls_name}/{specie_name}'
        os.makedirs(save_path, exist_ok=True)

        # Visualisierung
        #vis = cv2.cvtColor(cv2.resize(cv2.imread(img_path), (args.image_size, args.image_size)), cv2.COLOR_BGR2RGB)
        filetype = "png"
        mask = normalize(anomaly_map)
        saveHeatmap(save_path,
                    mask,
                    f"{img_path[-7:-4]}_hm",
                    filetype,
                    )

        # Speichere die Ergebnisse
        info_img = dict(
            img_path=f'{cls_name}/test/{specie_name}/{img_path[-7:]}',
            mask_path=f'{cls_name}/ground_truth/{specie_name}/{mask_path[-12:]}' if specie_name not in ['good', 'Good'] else '',
            hm_path=f'heatmaps/{cls_name}/{specie_name}/{img_path[-7:-4]}_hm.{filetype}',
            cls_name=cls_name,
            specie_name=specie_name,
            anomaly=anomaly,
            pred_label=int(np.squeeze(pred_label.cpu().numpy())),
            pr_sp=pr_sp,
        )

        if cls_name not in test_info['test_results'].keys():
            test_info['test_results'][cls_name] = []
        
        test_info['test_results'][cls_name].append(info_img)

    end_time = time.perf_counter()

    test_info['times']['time_preprocess'] = end_time_preprocess - start_time_preprocess
    test_info['times']['time'] = end_time - start_time  

    # Speichere die Ergebnisse
    test_results_path = f'{args.save_path}/{args.category}_test_results.json'
    print(os.path.abspath(test_results_path))
    with open(test_results_path, 'w') as f:
        f.write(json.dumps(test_info, indent=4) + '\n')
    print(f"saved test results to {test_results_path}")


def train(args):
    # Setup device and engine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Lade das Modell
    try:
        config = OmegaConf.load(args.model_config)
        model = get_model(config.model)
    except Exception as e:
        print("could not load config")
        print(args.model_config)
        print(e)
        model = get_model(model=args.model_name) #, config_path=args.model_config)
    model.to(device)

    # Lade das Datamodul

    # Definiere image_size (z. B. von args.image_size, als Liste [H, W])
    image_size = [args.image_size, args.image_size] #if hasattr(args, "image_size") else [256, 256]
    # datamodule = get_datamodule(dataset_name=args.dataset_name, dataset_path=args.dataset_path, batch_size=args.batch_size)
    data_config = DictConfig({"data": {"class_path": "MVTecAD",         
                                  "init_args": {"root": args.dataset_path,
                                                "category": args.category,
                                                },  
                                  "transform": {"image_size": image_size,
                                               },
                                 }
                        })
    print("\nimage Size: ", image_size, '\n')
    
    dataAugmentation = False
    if dataAugmentation: 
        # Nachträgliches Hinzufügen von Transforms
        train_transforms = [
            {"class_path": "albumentations.augmentations.transforms.HorizontalFlip", "init_args": {"p": 0.5}},
            {"class_path": "albumentations.augmentations.transforms.RandomBrightnessContrast", "init_args": {"p": 0.2}},
            {"class_path": "albumentations.augmentations.transforms.Rotate", "init_args": {"limit": 15, "p": 0.3}},
        ]

        data_config.data.transform.train = train_transforms

        # Optional: für Validierung/Test (meist keine Augmentierung)
        val_transforms = [
            # z.B. nur Resize/Normalization
        ]
        data_config.data.transform.val = val_transforms

    datamodule = get_datamodule(config=data_config)
    if hasattr(config, "dataset"):
        for key, value in config.dataset.items():
            if hasattr(datamodule, key):
                setattr(datamodule, key, value)

    # Erstelle die Engine
    logger_save_path = os.path.join(args.save_path, args.category, "logs")
    os.makedirs(logger_save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = TensorBoardLogger(save_dir=logger_save_path, 
                               name="anomalib_model", 
                               version=timestamp,
                               )

    callbacks = [GraphLogger, LossLoggerCallback()]
    if hasattr(config, 'trainer'):
        config.trainer.accelerator = device
        engine = Engine(#logger=logger, 
                        #callbacks=callbacks, 
                        **config.trainer)
    else:
        engine = Engine(max_epochs=args.epochs, 
                        accelerator=device, 
                        #logger=logger, 
                        #callbacks=callbacks
                        )

    start_time = time.perf_counter()
    # Trainiere das Modell

    engine.fit(model=model, datamodule=datamodule)
    
    end_time = time.perf_counter()
    duration = end_time-start_time
    print(f"Training Time: {duration:.2f} seconds, {duration/60:.2f} minutes")

    model_save_path = os.path.join(args.save_path, args.category)
    os.makedirs(model_save_path, exist_ok=True)
    # Speichere das Modell
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(model_save_path, f"{args.category}_{timestamp}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def define_parsers():
    # Schritt 1: Zuerst nur --mode parsen
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Switch mode between training (<train>) or testing (<test>)')
    initial_parser.add_argument('--seed', type=int, default=42, help='seed to set models on.')   
    mode_args, remaining_argv = initial_parser.parse_known_args()

    # Schritt 2: Neues Parser-Objekt mit spezifischen Argumenten
    parser = argparse.ArgumentParser(description='Test and train anomaly detection models')


    parser.add_argument('--mode', type=str, default=mode_args.mode, choices=['train', 'test'], help='Switch mode between training (<train>) or testing (<test>)')
    parser.add_argument('--seed', type=int, default=42, help='seed to set models on.')  

    setup_seed(mode_args.seed)
    if mode_args.mode == 'train':
        # Trainingsspezifische Argumente
        parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. padim, mahalanobis)')
        parser.add_argument('--model_config', type=str, required=False, help='Path to model config file')
        parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
        parser.add_argument('--save_path', type=str, required=True, help='Directory to save the trained model')
        parser.add_argument('--category', type=str, required=True, help='category to train on')
        parser.add_argument('--dataset_path', type=str, default='../data/dataset_masterproject', help='Path to test dataset')
        parser.add_argument("--image_size", type=int, default=256,
                            help="Input image size as [height width], e.g. --image_size 256. Default: 256x256")


    elif mode_args.mode == 'test':
        # Testspezifische Argumente
        parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g. padim, mahalanobis)')
        parser.add_argument('--model_config', type=str, required=True)
        parser.add_argument('--dataset_path', type=str, default='../data/dataset_masterproject', help='Path to test dataset')
        parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
        parser.add_argument('--save_path', type=str, default='./results/test_ocm', help='Path to save results')
        parser.add_argument('--image_size', type=int, default=256, help='Image size for visualization')
        parser.add_argument('--alpha', type=float, default=0.5, 
                            help='blending factor between original image and anomaly heatmap (0 = only heatmap, 1 = only image')
        parser.add_argument('--category', type=str, required=True, help='category to test model on')

    # Jetzt endgültig parsen
    args = parser.parse_args(remaining_argv)

    return args, mode_args

if __name__ == '__main__':
    # initial_parser = argparse.ArgumentParser('Test and train anomaly detection models')

    args, mode_args = define_parsers()

    if mode_args.mode == "train":
        train(args)
    elif mode_args.mode == "test":
        test(args)