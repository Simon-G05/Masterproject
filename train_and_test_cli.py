import sys
import argparse
import os
from datetime import datetime

# Importiere die Funktionen aus dem Hauptskript
from test_ocm_anomalib import train, test, define_parsers
from metrics import compute_metrics, define_parser

def get_latest_model_path(model_dir, category):
    # Liste alle Dateien, die mit category anfangen und auf .pth enden
    model_files = [f for f in os.listdir(model_dir)
                   if f.startswith(category) and f.endswith(".pth")]

    if not model_files:
        return None  # Kein Modell gefunden

    # Extrahiere Timestamp aus Dateinamen und sortiere danach
    def extract_timestamp(filename):
        # Beispiel filename: bottle_20250519_143015.pth
        # Extrahiere Teil zwischen "_" und ".pth"
        ts_str = filename[len(category)+1:-4]  # "20250519_143015"
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")

    # Sortiere Dateien nach Timestamp, neueste zuerst
    model_files.sort(key=extract_timestamp, reverse=True)

    # Gib kompletten Pfad zur neuesten Datei zurück
    return os.path.join(model_dir, model_files[0])

def parse_categories():
    parser = argparse.ArgumentParser()
    parser.add_argument('--categorys', nargs='+', required=True, help='Liste der Kategorien')
    args, unknown = parser.parse_known_args()
    return args.categorys

if "__main__" == __name__:
    categorys = parse_categories()

    # category = 'bottle'
    for category in categorys:
        model = 'padim'
        model_config = '/workspace/Master/Padim.yaml'
        modelBaseDir = '/workspace/Master/results/train_ocm'
        test_dir = '/workspace/Master/results/test_ocm_padim'
        path_dataset = '/workspace/Master/data/MVTecAD'
        img_size = '512'

        # Simuliere die Kommandozeilenargumente für das Training
        sys.argv = [
            'train_and_test.py',  # Name des Skripts
            '--mode', 'train',
            '--model_name', model,
            '--model_config', model_config,
            '--epochs', '1',
            '--dataset_path', path_dataset,
            '--save_path', modelBaseDir,
            '--image_size', img_size,
            '--category', category,
        ]

        args, mode = define_parsers()

        train(args)

        model_dir = f'{modelBaseDir}/{category}'
        model_path = get_latest_model_path(model_dir, category)
        if model_path is None:
            raise ValueError("could not find model")

        sys.argv = [
            'train_and_test.py',  # Name des Skripts
            '--mode', 'test',
            '--model_name', model,
            '--model_config', model_config,
            '--dataset_path', path_dataset,
            '--save_path', test_dir,
            '--image_size', img_size,
            '--category', category,
            '--checkpoint_path', model_path,
            '--alpha', '0',
        ]

        args, mode = define_parsers()

        test(args)

        sys.argv = [
            'train_and_test.py',  # Name des Skripts
            '--dataset_path', path_dataset,
            '--test_results_path', test_dir,
            '--save_path', test_dir,
            '--category', category,
        ]

        compute_metrics(define_parser())