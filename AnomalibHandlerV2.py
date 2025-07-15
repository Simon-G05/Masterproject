import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from anomalib.data import MVTecAD
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.loggers import AnomalibMLFlowLogger, AnomalibTensorBoardLogger, AnomalibWandbLogger

from omegaconf import OmegaConf
import os
from pathlib import Path
import numpy as np
import cv2
import json

def tryLoad(obj, attr_name):
    try:
        return getattr(obj, attr_name)
    except Exception:
        print(f"could not load persistent {attr_name}")
        return None

class AnomalyPipeline:
    def __init__(self, useGpu: bool = False):
        self.useGpu = "auto" if useGpu else "cpu"
    
    def createDatamodule(self, 
                         pathDataset: str, 
                         category: str,
                         batchSize: int = 32, 
                         numWorkers: int = 8,
                         persistent: bool = False,
                         ):
        
        if not os.path.isdir(pathDataset):
            raise Exception("Dataset dir does not exist")
        
        datamodule = MVTecAD(
            category=category,
            root=pathDataset,
            train_batch_size=batchSize,
            eval_batch_size=batchSize,
            num_workers=numWorkers
        )
        if persistent:
            self.datamodule = datamodule

        return datamodule
    
    def createModel(self,
                    pathConfigFile: str,
                    model = Padim,
                    persistent: bool = False,
                    ):
        
        if not os.path.isfile(pathConfigFile):
            raise Exception("Configfile does not exist")

        config = OmegaConf.load(pathConfigFile)
        
        model = model(**config.model.init_args)

        if persistent:
            self.model = model 

        return model

    def createEngine(self,
                     maxEpochs: int = 10,
                     useGpu: bool = None,
                     persistent: bool = False,
                     logger = None,
                     callbacks = None,
                     defaultRootDir = "results/models"
                     ):

        useGpu = self.useGpu if useGpu is None else useGpu

        engine = Engine(max_epochs=maxEpochs,
                        accelerator=useGpu,
                        devices=1,
                        callbacks=callbacks,
                        logger=logger,
                        default_root_dir=defaultRootDir,
                        )
        
        if persistent:
            self.engine = engine

        return engine
    
    def trainOneClassModel(self, 
                           datamodule = None,
                           model = None,
                           engine: Engine = None,
                           usePersistent: bool = False,       
                           ):
        if usePersistent:
            engine = tryLoad(self, "engine")
            model = tryLoad(self, "model")
            datamodule = tryLoad(self, "datamodule")

        else:
            if model is None or engine is None or datamodule is None:
                raise Exception("model, engine or datamodule is None")
            
        engine.fit(model=model,
                   datamodule=datamodule)

    def loadModel(self, 
                  category: str, 
                  modelType = Padim,
                  modelpath: str = None,
                  modelversion: str = "latest",
                  persistent: bool = False,                  
                  ):

        if modelpath is not None:
            path = Path(modelpath) / category / modelversion / "weights" / "lightning" / "model.ckpt"
            # Lade dein trainiertes Modell aus Checkpoint
            model = modelType.load_from_checkpoint(path)
            if persistent:
                self.model = model
            return model
        else:
            tryLoad(self, "model")

    def testModel(self, 
                  engine = None,
                  datamodule = None,
                  modelpath: str = None, 
                  modelversion: str = "latest"
                  ):
        if datamodule is None:
            datamodule = tryLoad(self, "datamodule")
        if engine is None:
            engine = tryLoad(self, "engine")

        category = datamodule.category

        # 1. Lade das aktuell in der Klasse gespeicherte Modell oder die ckpt datei
        model = self.loadModel(category, 
                               modelpath=modelpath, 
                               modelversion=modelversion)
        # 2. Setze das Modell in den Eval-Modus
        model.eval()
        # 3. Berechne die Scores
        return engine.test(model, datamodule=datamodule)

    def predictCategory(self,     
                        engine = None,                   
                        datamodule = None,
                        model = None, 
                        ):
        if datamodule is None:
            datamodule = tryLoad(self, "datamodule")
        if engine is None:
            engine = tryLoad(self, "engine")
        if model is None:
            model = tryLoad(self, "model")
        
        category = datamodule.category
        # 1. Lade das aktuell in der Klasse gespeicherte Modell oder die ckpt datei

        # 2. Setze das Modell in den Eval-Modus
        model.eval()
        # 3. Berechne die Scores
        return engine.predict(model, datamodule=datamodule)
    
    def saveHeatmap(self,
                    pathOut: str,
                    heatmap: list,
                    name: str,
                    fileType: str = "png"
                    ):
        if heatmap.dtype != np.float32 and heatmap.dtype != np.float64:
            raise ValueError("Heatmap must be float32 or float64")

        if not ((0 <= heatmap).all() and (heatmap <= 1).all()):
            raise ValueError("Heatmap values must be in the range [0, 1]")
                
        pathFile = Path(pathOut) / f"{name}.{fileType}"
        
        # Skaliere auf 16-Bit
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        os.makedirs(pathOut, exist_ok=True)
        # Speichern als .tif
        cv2.imwrite(pathFile, heatmap_uint8)

        return str(pathFile)

    def savePredictions(self, 
                           predictions,
                           pathOutput,
                           timeTotal,
                           timePreprocess,
                           fileName: str = "predictions.json"
                           ) -> list:
        """
        Speichert Anomalib-Vorhersagen in einer JSON-Datei im gewünschten Format.

        Args:
            predictions (list[dict]): Liste der Prediction-Dictionaries.
            pathOutput (str): Pfad zur Ausgabedatei (JSON).
            timeTotal (float): Gesamtzeit der Inferenz.
            timePreprocess (float): Zeit für Preprocessing.
        """

        output_data = {
            "times": {
                "time_preprocess": timePreprocess,
                "time": timeTotal
            },
            "test_results": {}
        }
        for instance in predictions:
            for pred in instance:
                Dirs = pred.image_path.split("/")
                cls_name = Dirs[-4]
                path = "/".join(Dirs[-4:-1])
                hmPath = self.saveHeatmap(f"{pathOutput}/heatmaps/{path}", 
                                          pred.anomaly_map.numpy(), 
                                          Dirs[-1].split(".")[0])
                absoluteHmPath = Path(hmPath).resolve()
                output_data["test_results"].setdefault(cls_name, []).append({
                    "img_path": pred.image_path,
                    "mask_path": pred.mask_path,
                    "hm_path": str(absoluteHmPath),
                    "cls_name": cls_name,
                    "specie_name": Dirs[-2],
                    "anomaly": int(pred.pred_label),
                    "pr_sp": float(pred.pred_score)
                })

        Path(pathOutput).mkdir(parents=True, exist_ok=True)
        filePath = Path(pathOutput) / fileName

        with open(filePath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)

if "__main__" == __name__:
    ah = AnomalyPipeline(True)
    pathDataset = "/Volumes/Extreme SSD/Master/Projekt/datasets/MVTecAD"
    datamodule = ah.createDatamodule(pathDataset, "bottle")
    pathConfig = "/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/Programme/OneClass/configFiles/padim.yaml"
    model = ah.createModel(pathConfig)
    # # logger = AnomalibTensorBoardLogger(save_dir="logs/tensorboard", log_graph=True)
    engine = ah.createEngine()
    # ah.trainOneClassModel(datamodule, model, engine, False)

    modelpath = "/Users/simon/Documents/HS_Kempten/Projekt/Git/results/models/Padim/MVTecAD"
    model = ah.loadModel("bottle", 
                         modelpath=modelpath, 
                         modelversion="latest")
    # model.eval()
    predictions = ah.predictCategory(engine, datamodule, model)

    ah.savePredictions(predictions, "results/predictions", "10s", "10s", fileName="bottle.json")
