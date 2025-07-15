import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.brain as fob # ML methods
import fiftyone.zoo as foz # zoo datasets and models
import glob
import os
import cv2
import numpy as np
import re
from pathlib import Path
from enum import Enum, auto
import json
from PIL import Image

class FieldTypes(Enum):
    Mask = auto(),
    Label = auto(),
    Heatmap = auto(),

class FOVisualizer:
    def __init__(self, colorBy:str = "field"):
        self.debugOut = True

        defaultColorPool = [
                "#EE0000",  # rot
                "#EE6600",  # orange
                "#993300", 
                "#996633",
                "#999900",  # gelb
                "#009900",  # grün
                "#003300",  # dunkelgrün
                "#009999",
                "#000099",  # dunkelblau
                "#0066FF",  # hellblau
                "#6600FF",  # lila
                "#CC33CC",  # pink
                "#777799",
            ]
        
        AllowedColorBy = ["value", "field", "instance"]
        if colorBy not in AllowedColorBy:
            raise ValueError(f"colorBy must be one of the following: {AllowedColorBy}")
        fo.app_config.color_by = colorBy

        self.colorScheme = fo.ColorScheme(
            color_by = colorBy,
            color_pool = defaultColorPool,
        )


    def str2list(self, string: str) -> list:
        """changes object of type str in object of type list"""

        return [string] if isinstance(string, str) else string

    def loadDataset(self, name: str) -> fo.Dataset:
        """trys to load dataset from disc"""

        try:
            return fo.load_dataset(name)
        except Exception as e:
            print(f"Loading Dataset {name} not possible")
            if self.debugOut:
                print("Error: ", e)
            return None

    def deleteDataset(self, name: str | list) -> None:
        """If Dataset *name* exists dataset is deleted"""

        datasets = self.str2list(name)

        for dataset in datasets:
            try:
                fo.delete_dataset(dataset)
                print(f"dataset {dataset} has been deleted")
            except Exception as e:
                print(f"dataset {dataset} could not be deleted")
                if self.debugOut:
                    print(f"Error: {e}")
    
    def deleteAllDatasets(self) -> None:
        """Deletes all existing FiftyOne datasets"""

        sets = self.listDatasets()
        self.deleteDataset(sets)
        
    def listDatasets(self) -> list:
        """returns a list of all existing FiftyOne Datasets"""
        return fo.list_datasets()
    
    def AddLabel2Sample(self, sample: fo.Sample, 
                        labelname: str | list, label: str | list) -> fo.Sample:
        """
        add one or more classification labels to a sample. 
            
        :param sample: sample to which the label is added

        :param labelname: name of the label field to be added (str or list of str)

        :param label: classification label(s) to be added (str or list of str)

        :return: the modified sample with the label(s) added
        """

        names = self.str2list(labelname)
        labels = self.str2list(label)

        if len(labels) != len(names):
            raise ValueError(f"Number of passed labelnames ({len(names)}) must ", 
                             f"be equal to number of passed labels ({len(labels)})")

        for name, label in zip(names, labels):
            sample[name] = fo.Classification(label=label)

        return sample
    
    def AddMask2Sample(self, sample: fo.Sample,
                       maskname: str | list, mask: list) -> fo.Sample:
        """
        add one or more masks to a sample. 
            
        :param sample: sample to which the mask is added

        :param maskname: name of the mask to be added (str or list of str)

        :param mask: value of the labels to be added (as a list of masks)

        :return: the modified sample with the mask(s) added
        """

        names = self.str2list(maskname)
        if len(names) == 1:
            mask = [mask]

        if len(names) != len(mask):
            raise ValueError(f"Number of passed masknames ({len(names)}) must ",
                             f"be equal to number of passed masks ({len(mask)})")

        for name, mask in zip(names, mask):
            sample[name] = fo.Segmentation(mask=mask)

        return sample
    
    def binaryMaskFromImage(self, mask: list = None, img_path : str = None) -> list:
        """
        convert a grayscale image or mask into a binary mask.

        :param mask: grayscale mask as a list or numpy array (optional)

        :param img_path: path to the grayscale image file (optional)

        :return: binary mask as a list where pixel values are either 0 or 1
        """

        if mask is None and img_path is None:
            raise ValueError("Either 'mask' or 'img_path' must have a value other than None")
        
        if mask is None:
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        return (mask > 128).astype(np.uint8)

    def AddHeatmap2Sample(self, sample: fo.Sample, 
                          hmname: str | list, heatmap: list,
                          normalize: bool = False, minMaxAsRange: bool = False,
                          min: str | float = 0, max: str | float = 1) -> fo.Sample:
        """
        add one or more heatmaps to a sample.
        
        :param sample: sample to which the heatmap is added

        :param hmname: name of the heatmap field to be added (str or list of str)

        :param heatmap: list of heatmap arrays to be added

        :param normalize: whether to normalize the heatmap values to [0, 1] (default: False)

        :param minMaxAsRange: whether to use the min and max of each heatmap as its display range (default: False)

        :param min: minimum value of the display range if not using minMaxAsRange (default: 0)

        :param max: maximum value of the display range if not using minMaxAsRange (default: 1)

        :return: the modified sample with the heatmap(s) added
        """

        names = self.str2list(hmname)
        if len(names) == 1:
            maps = [heatmap]

        if len(names) != len(maps):
            raise ValueError(f"Number of passed heatmapnames ({len(names)}) must ",
                             f"be equal to number of passed heatmaps ({len(maps)})")
        
        for name, map in zip(names, maps):
            if normalize:
                # Normalize assuming input is in 0–255 range
                map = map.astype("float32") / 255.0
                min_val = 0.0
                max_val = 1.0
            elif minMaxAsRange:
                min_val = map.min()
                max_val = map.max()
            else:
                min_val = float(min)
                max_val = float(max)

        
            sample[name] = fo.Heatmap(map=map, range=[min_val, max_val])

        return sample
    
    def addReplaceField(self, fields: list, new_field: dict, key: str = "path") -> list:
        """
        replace an existing dictionary in a list by matching a key, or add it if not present.

        :param fields: list of existing dictionaries (fields)

        :param new_field: dictionary to add or replace in the list

        :param key: key used to identify which dictionary to replace (default: "path")

        :return: updated list of dictionaries with the new_field added or replaced
        """
     
        path_to_replace = new_field.get(key)

        # Filtere alle alten Einträge mit dem gleichen path raus
        fields = [f for f in fields if f.get(key) != path_to_replace]

        # Füge den neuen Eintrag hinzu
        fields.append(new_field)

        return fields

    def AddField2ColorScheme(self, fieldname: str, fieldColor: str, 
                             fieldType: FieldTypes, 
                             colorScheme: fo.ColorScheme = None,
                             valueColors: dict | list = None) -> fo.ColorScheme:
        """
        add a field to a color scheme configuration for visualization.

        :param fieldname: name of the field to be added

        :param fieldColor: base color to associate with the field

        :param fieldType: type of the field (e.g., Label, Mask)

        :param colorScheme: existing color scheme to update (optional)

        :param valueColors: specific value-to-color mappings (dict for labels, list for default coloring)

        :return: the updated ColorScheme with the new field configuration added
        """
        
        field = {
            "path": fieldname,
            "fieldColor": fieldColor,
        }
        
        if not isinstance(fieldType, FieldTypes):
            raise ValueError(f"fieldType must be of type Fieldtypes not {type(fieldType)}")

        if fieldType == FieldTypes.Label:
            colors = []
            if isinstance(valueColors, dict) and fieldType == FieldTypes.Label: 
                field["colorByAttribute"] = "label"  
                for key in valueColors.keys:
                    valCol = {"value": key, "color": valueColors[key]}
                    colors.append(valCol)
                field["valueColors"] = colors
            elif isinstance(valueColors, list):
                for value in valueColors:
                    valCol = {"value": value, "color": fieldColor}
                    colors.append(valCol)
                field["valueColors"] = colors
            elif valueColors is not None:
                raise ValueError("valueColors must be of type dict or list.",
                                f"Current type: {type(valueColors)}")
        elif fieldType == FieldTypes.Mask:
            field["maskTargetsColors"] = [
                    {"intTarget": 1, "color": fieldColor}
                ]
            field["colorByAttribute"] = "field" 
        else:
            field["colorByAttribute"] = "value"  

        if colorScheme is None:
            colorScheme = self.colorScheme
        if colorScheme.fields is not None:
            fields =self.addReplaceField(colorScheme.fields, field)
        else:
            fields = [field]

        colorScheme.fields = fields
        return colorScheme

    def updateColorscheme(self, colorScheme: fo.ColorScheme) -> None:
        """
        update the current color scheme.

        :param colorScheme: the new ColorScheme object to set
        """        
        
        self.colorScheme = colorScheme

    def applyColorScheme2Dataset(self, dataset: fo.Dataset, 
                                 colorScheme: fo.ColorScheme = None) -> fo.Dataset:
        """
        apply a color scheme to a dataset configuration.

        :param dataset: the dataset to which the color scheme will be applied

        :param colorScheme: the ColorScheme object to apply (optional, defaults to the current color scheme)

        :return: the updated dataset with the applied color scheme
        """
        
        if colorScheme is None:
            colorScheme = self.colorScheme
        
        dataset.app_config.color_scheme = colorScheme
        dataset.save()
        return dataset

    def setDefaultColorscales(self, colorscale: str,
                              colorScheme: fo.ColorScheme = None) -> fo.ColorScheme:
        """
        set the default colorscale for a color scheme.

        :param colorscale: the name of the colorscale to set as default

        :param colorScheme: the ColorScheme object to update (optional, defaults to the current color scheme)

        :return: the updated ColorScheme with the default colorscale set
        """
        
        if colorScheme is None:
            self.colorScheme.default_colorscale = {"name": colorscale, "list": None}
            return self.colorScheme
        else:
            colorScheme.default_colorscale = {"name": colorscale, "list": None}
            return colorScheme

    def setSpecificColorscale(self, fieldname: str, colorscale: str,
                              colorScheme: fo.ColorScheme = None) -> fo.ColorScheme:
        """
        set a specific colorscale for a field in the color scheme.

        :param fieldname: the name of the field for which to set the colorscale

        :param colorscale: the name of the colorscale to apply to the field

        :param colorScheme: the ColorScheme object to update (optional, defaults to the current color scheme)

        :return: the updated ColorScheme with the specific colorscale set for the field
        """        
        
        if colorScheme is None:
            colorScheme = self.colorScheme
        
        scale = {"path": fieldname,
                 "name": colorscale
                 }
        self.addReplaceField(colorScheme.colorscales, scale)

        return colorScheme
        
    def createDatasetFromDirStruct(self, path: str, name: str,
                            category: str | list = "*", 
                            trainTestSplit: str | list = "*",
                            defect: str | list = "*",
                            imgFileType: str = "png",
                            overwrite: bool = True,
                            mask_dir: str = None
                            ) -> fo.Dataset:
        """
        create a dataset from a directory structure containing images and optional masks.

        :param path: the root directory path where the dataset files are located

        :param name: the name to assign to the dataset

        :param category: list of categories to include (optional, "*" includes all)

        :param trainTestSplit: list of train/test splits to include (optional, "*" includes all)

        :param defect: list of defect types to include (optional, "*" includes all)

        :param imgFileType: the image file type (default is "png")

        :param overwrite: whether to overwrite the dataset if it already exists (default is True)

        :param mask_dir: the subdirectory containing the mask images (optional)

        :return: the created Dataset object containing samples from the directory structure
        """        
        
        # Filterlisten vorbereiten
        cat_filter = set(self.str2list(category)) if category else None
        split_filter = set(self.str2list(trainTestSplit)) if trainTestSplit else None
        defect_filter = set(self.str2list(defect)) if defect else None

        dataset = fo.Dataset(name=name, overwrite=overwrite)
        root = Path(path)

        samples = []
        for file in root.rglob(f"*.{imgFileType}"):
            parts = file.parts[-4:]  # erwartet: category/split/defect/file.png
            if len(parts) < 4:
                continue

            cat_sample, split_sample, defect_sample = parts[0], parts[1], parts[2]

            # Filter anwenden, wenn gesetzt
            if cat_filter and "*" not in cat_filter and cat_sample not in cat_filter:
                continue
            if split_sample == mask_dir:
                continue
            if split_filter and "*" not in split_filter and split_sample not in split_filter:
                continue
            if defect_filter and "*" not in defect_filter and defect_sample not in defect_filter:
                continue

            # Sample erstellen
            sample = fo.Sample(
                filepath=str(file),
                tags=[split_sample],
                metadata=None,
                Category=fo.Classification(label=cat_sample),
                Defect=fo.Classification(label=defect_sample),
            )

            if mask_dir:
                filename = re.sub(rf"\.{imgFileType}$", f"_mask.{imgFileType}", parts[3], flags=re.IGNORECASE)
                gt_path = Path(path) / cat_sample / mask_dir / defect_sample / filename
                if gt_path.exists():
                    binary_mask = self.binaryMaskFromImage(img_path=gt_path)
                    self.AddMask2Sample(sample, "ground_truth", binary_mask)

            samples.append(sample)
        
        dataset.add_samples(samples)
        return dataset

    def datasetFromAnomalibPreds(self, predictions: list, name: str, addCategory:bool = False) -> fo.Dataset:
        """
        create a dataset from anomaly detection predictions.

        :param predictions: list of prediction sets, where each set contains prediction objects with fields 
                            like image_path, pred_mask, gt_mask, pred_label, gt_label, and anomaly_map.

        :return: the created Dataset object containing samples from the predictions
        """
        
        #custom_path = "/Volumes/Extreme SSD/Master/Projekt/datasets/fiftyone"
        dataset = fo.Dataset(name=name, overwrite=True, 
                             persistent=True)

        for set in predictions:
            for pred in set:
                image_path = pred.image_path
                pred_mask = pred.pred_mask.cpu().numpy().astype(np.uint8)
                
                if hasattr(pred, "gt_mask") and pred.gt_mask is not None:
                    gt_mask = pred.gt_mask.cpu().numpy().astype(np.uint8)
                else:
                    gt_mask = np.zeros_like(pred_mask)

                sample = fo.Sample(filepath=image_path)
                
                sample.compute_metadata()

                # predicted mask
                self.AddMask2Sample(sample, "prediction", pred_mask)
                # Ground Truth Maske
                self.AddMask2Sample(sample, "ground_truth", gt_mask)

                pred_label = "Anomaly" if pred.pred_label else "Good"
                gt_label = "Anomaly" if pred.gt_label else "Good"
                self.AddLabel2Sample(sample, ["pred_label", "gt_label"], [pred_label, gt_label])

                if addCategory:
                    self.AddLabel2Sample(sample, "category", name)

                raw_map = pred.anomaly_map.numpy()
                self.AddHeatmap2Sample(sample, "pred_heatmap", raw_map, False, False)

                dataset.add_sample(sample)
        return dataset

    def datasetFromJSON(self, 
                        filepath, 
                        category: str, 
                        name: str,
                        anomalyThreshold: float,
                        sourcePathHM: str = None,
                        sourcePathSet: str = None,
                        ):
        
        dataset = fo.Dataset(name=name, overwrite=True, 
                             persistent=True)
        
        with open(filepath, "r") as f:
            data = json.load(f)

        for sample in data["test_results"][category]:
            imgPath = Path(sample["img_path"]) if sample["img_path"] is not None else Path("None")
            gtMaskPath = Path(sample["mask_path"]) if sample["mask_path"] is not None else Path("None")
            hmPath = Path(sample["hm_path"]) if sample["hm_path"] is not None else Path("None")
            gtCategory = sample["cls_name"]
            gtLabel = sample["specie_name"]
            predLabel = sample["anomaly"]
            predScore = sample["pr_sp"]

            if sourcePathHM is not None:
                hmPath = Path(sourcePathHM) / hmPath
            if sourcePathSet is not None:
                imgPath = Path(sourcePathSet) / imgPath
                gtMaskPath = Path(sourcePathSet) / gtMaskPath

            # Überprüfen, ob die Dateien existieren
            files_exist = {
                "image": imgPath.exists() and imgPath._str != sourcePathSet,
                "heatmap": hmPath.exists() and hmPath._str != sourcePathHM,
                "gt_mask": gtMaskPath.exists() and gtMaskPath._str != sourcePathSet,
            }

            # create sample
            if files_exist["image"]:
                # Bild mit PIL öffnen und Größe abfragen
                with Image.open(imgPath) as img:
                    width, height = img.size
                foSample = fo.Sample(filepath=imgPath)
            else: 
                continue

            if files_exist["heatmap"]:
                hm = cv2.imread(hmPath, cv2.IMREAD_GRAYSCALE)
                self.AddHeatmap2Sample(foSample, "predHeatmap", hm, normalize=True)

                predMask = (hm >= anomalyThreshold * 255).astype(np.uint8)
                self.AddMask2Sample(foSample, "prediction", predMask)
            
            # add Ground Truth mask
            if files_exist["gt_mask"]:
                binary_mask = self.binaryMaskFromImage(img_path=gtMaskPath)
            else:
                binary_mask = np.zeros((width, height))
            self.AddMask2Sample(foSample, "groundTruth", binary_mask)
            
            # add Labels
            self.AddLabel2Sample(foSample, ["categroy", "gtLabel"], [gtCategory, gtLabel])

            predLabel = "Anomaly" if predLabel > 0.5 else "Good"
            foSample["predLabel"] = fo.Classification(label=predLabel, confidence=predScore)

            foSample.compute_metadata()

            dataset.add_sample(foSample)

        return dataset

    def mergeDatasets(self, name: str, datasets: list, deleteOriginalSets: bool = False):
        merged_dataset = fo.Dataset(name=name, overwrite=True, persistent=True)

        for datasetName in datasets:
            dataset = fo.load_dataset(datasetName)
            merged_dataset.add_collection(dataset, new_ids=True)
            if deleteOriginalSets:
                self.deleteDataset(dataset)
        return merged_dataset

    def defaultColoration(self, dataset: fo.Dataset,
                          namePredLabel: str = "pred_label",
                          nameGtLabel: str = "gt_label",
                          namePredMask: str = "prediction",
                          nameGtMask: str = "ground_truth",
                          namePredHm: str = "pred_heatmap",
                          gtLabels: list = ["Anomaly", "Good"],
                          predLabels: list = ["Anomaly", "Good"],
                          colorGt: str = "green",
                          colorPred: str = "red",
                          colorscale: str = "jet",
                          ):
        vis.AddField2ColorScheme(namePredLabel, colorPred, FieldTypes.Label, valueColors=predLabels)
        vis.AddField2ColorScheme(nameGtLabel, colorGt, FieldTypes.Label, valueColors=gtLabels)

        vis.AddField2ColorScheme(namePredMask, colorPred, FieldTypes.Mask)
        vis.AddField2ColorScheme(nameGtMask, colorGt, FieldTypes.Mask)

        vis.AddField2ColorScheme(namePredHm, colorPred, FieldTypes.Heatmap)
        vis.setDefaultColorscales(colorscale)

        vis.applyColorScheme2Dataset(dataset)

if "__main__" == __name__:
    vis = FOVisualizer("value")
    # print(vis.listDatasets())
    # data = vis.mergeDatasets("merged", vis.listDatasets())

    # vis.AddField2ColorScheme("pred_label", "red", FieldTypes.Label, valueColors=["Anomaly", "Good"])
    # vis.AddField2ColorScheme("gt_label", "green", FieldTypes.Label, valueColors=["Anomaly", "Good"])

    # vis.AddField2ColorScheme("prediction", "red", FieldTypes.Mask)
    # vis.AddField2ColorScheme("ground_truth", "green", FieldTypes.Mask)

    # vis.AddField2ColorScheme("pred_heatmap", "red", FieldTypes.Heatmap)
    # vis.setDefaultColorscales("jet")

    # vis.applyColorScheme2Dataset(data)

    # datasets = fo.list_datasets()
    # data = vis.mergeDatasets("Test", datasets)
    # vis.defaultColoration(data)
    # print(fo.list_datasets())
    # session = fo.launch_app(data)
    # session.wait()


    # dataset = fov.createDatasetFromDirStruct(path="/Volumes/Extreme SSD/Master/Projekt/datasets/MVTecAD",
    #                                     name="Test", category="bottle", trainTestSplit=None,
    #                                     mask_dir="ground_truth")
    # fov.AddField2ColorScheme("Category", "pink", FieldTypes.Label)
    # fov.AddField2ColorScheme("Defect", "orange", FieldTypes.Label)
    # fov.AddField2ColorScheme("ground_truth", "red", FieldTypes.Mask)
    # fov.applyColorScheme2Dataset(dataset)

    # dataset = fov.loadDataset("Test")

    # print(dataset)
    # session = fo.launch_app(dataset)
    # session.wait()
    categorys = ['bottle', 'cable', 'grid', 'metal_nut', 'screw', 'wood']
    datasets = []
    sourceDir = '/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results'
    model = 'test_ocm_patchcore'

    for category in categorys:
        metricsJSON = os.path.join(sourceDir, model, f"{category}_metric_results.json")
        testJSON = os.path.join(sourceDir, model, f"{category}_test_results.json")
        with open(metricsJSON) as f_out:
            metrics = json.load(f_out)
        thr = metrics['metric_results'][category]['threshold']
        print(thr)
        data = vis.datasetFromJSON(testJSON, 
                                   category, 
                                   f"TestJson_{category}", 
                                   thr,
                                   sourcePathHM=os.path.join(sourceDir, model),
                                   sourcePathSet='/Volumes/Extreme SSD/Master/Projekt/datasets/MVTecAD',
                                   )
        datasets.append(f"TestJson_{category}")

    data = vis.mergeDatasets(model, datasets)
    print(fo.list_datasets())
    # vis.defaultColoration(data,
    #                       namePredLabel="predLabel",
    #                       nameGtLabel="gtLabel",
    #                       nameGtMask="groundTruth",
    #                       namePredHm="predHeatmap",
    #                       namePredMask="prediction",
    #                       gtLabels=["good", "broken_large", "broken_small", "contamination"]
    #                       )

    session = fo.launch_app(data)
    session.wait()
    session.close()