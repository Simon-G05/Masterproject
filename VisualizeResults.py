import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    f1_score, accuracy_score, precision_score, recall_score
)

class AnomalyEvaluator:
    def __init__(self, y_true, y_score, threshold=0.5, specie_true=None, save_path=None):
        self.y_true = np.array(y_true)
        self.y_score = np.array(y_score)
        self.threshold = threshold
        self.y_pred = (self.y_score >= threshold).astype(int)
        self.save_path = save_path  # Optionaler Pfad zum Speichern
        self.specie_true = specie_true

        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

    def calculate_yPred(self):
        self.y_pred = (self.y_score >= self.threshold).astype(int)

    def print_metrics(self):
        print(f"Accuracy:  {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"F1 Score:  {f1_score(self.y_true, self.y_pred):.4f}")
        print(f"Precision: {precision_score(self.y_true, self.y_pred):.4f}")
        print(f"Recall:    {recall_score(self.y_true, self.y_pred):.4f}")
        print(f"AUC:       {roc_auc_score(self.y_true, self.y_score):.4f}")
        print(f"AP:        {average_precision_score(self.y_true, self.y_score):.4f}")
    
    def plot_precision_recall_curve(self, filename="precision_recall_curve.png"):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        ap = average_precision_score(self.y_true, self.y_score)
        plt.figure()
        plt.plot(recall, precision, label=f'AP = {ap:.4f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_roc_curve(self, filename="roc_curve.png"):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
        auc = roc_auc_score(self.y_true, self.y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_confusion_matrix(self, filename="confusion_matrix.png"):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix (threshold = {self.threshold})")
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_confusion_matrix_custom(self, filename="custom_confusion_matrix.png"):
        # Alle Klassen, z. B. ['Defekt A', 'Defekt B', 'Defekt C', 'Normal']
        classes = sorted(set(self.specie_true))

        # Initialisiere Matrix
        matrix = {cls: {"Good": 0, "Anomaly": 0} for cls in classes}

        # Fülle Matrix
        for true_label, pred_label, class_label in zip(self.y_true, self.y_pred, self.specie_true):
            if pred_label == 1:
                matrix[class_label]["Anomaly"] += 1  # Defekt erkannt
            else:
                matrix[class_label]["Good"] += 1

        # In DataFrame umwandeln
        df = pd.DataFrame(matrix).T[["Anomaly", "Good"]].T  # Drehen für gewünschte Ansicht

        # Plot
        plt.figure(figsize=(10, 4))
        sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Custom Confusion Matrix (binary prediction vs. multiclass truth)")
        plt.ylabel("Vorhersage")
        plt.xlabel("Tatsächliche Klasse")
        plt.tight_layout()
        self._save_or_show(filename)


    def plot_score_distribution(self, filename="score_distribution"):
        plt.figure()
        sns.kdeplot(self.y_score[self.y_true == 0], label="Normal", fill=True, color="green")
        sns.kdeplot(self.y_score[self.y_true == 1], label="Anomal", fill=True, color="red")
        plt.axvline(self.threshold, color='red', linestyle='--', label='Threshold')
        plt.xlabel("Anomalie Score")
        plt.ylabel("Dichte")
        plt.title("Scoreverteilung")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_threshold_vs_metrics(self, filename="threshold_vs_metrics.png"):
        thresholds = np.linspace(0, 1, 100)
        f1s, precisions, recalls = [], [], []

        for t in thresholds:
            y_pred = (self.y_score >= t).astype(int)
            f1s.append(f1_score(self.y_true, y_pred))
            precisions.append(precision_score(self.y_true, y_pred, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred))

        plt.figure()
        plt.plot(thresholds, f1s, label="F1 Score", color="navy")
        plt.plot(thresholds, precisions, label="Precision", linestyle="-.", color="purple")
        plt.plot(thresholds, recalls, label="Recall", linestyle="-.", color="deeppink")
        plt.axvline(self.threshold, color='red', linestyle='--', label='Aktueller Threshold')
        plt.xlabel("Threshold")
        plt.ylabel("Metrik")
        plt.ylim(0, 1.05)
        plt.title("Threshold vs. Metriken")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_misclassification_by_threshold(self, filename="misclassification_by_threshold.png"):
        thresholds = np.linspace(0, 1, 200)
        normal_misrate, anomaly_misrate = [], []

        for t in thresholds:
            y_pred = (self.y_score >= t).astype(int)
            normal_mask = (self.y_true == 0)
            if normal_mask.sum() > 0:
                normal_errors = (y_pred[normal_mask] != 0).sum()
                normal_misrate.append(normal_errors / normal_mask.sum())
            else:
                normal_misrate.append(np.nan)

            anomaly_mask = (self.y_true == 1)
            if anomaly_mask.sum() > 0:
                anomaly_errors = (y_pred[anomaly_mask] != 1).sum()
                anomaly_misrate.append(anomaly_errors / anomaly_mask.sum())
            else:
                anomaly_misrate.append(np.nan)

        plt.figure(figsize=(8, 6))
        plt.plot(thresholds, normal_misrate, label="Fehler bei normalen Beispielen (FP)", color="green")
        plt.fill_between(thresholds, 0, normal_misrate, color="green", alpha=0.3)
        plt.plot(thresholds, anomaly_misrate, label="Fehler bei anomalen Beispielen (FN)", color="red")
        plt.fill_between(thresholds, 0, anomaly_misrate, color="red", alpha=0.3)
        plt.axvline(self.threshold, color='gray', linestyle='--', label="Aktueller Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Fehleranteil")
        plt.title("Fehlklassifikationsrate vs. Threshold")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        self._save_or_show(filename)

    def plot_all_metrics_combined(self, filename="combined_metrics.png"):
        max_val = max(self.y_score)
        thresholds = np.linspace(0, max_val, 200)
        f1s, precisions, recalls = [], [], []
        normal_misrate, anomaly_misrate = [], []

        for t in thresholds:
            y_pred = (self.y_score >= t).astype(int)
            f1s.append(f1_score(self.y_true, y_pred, zero_division=0))
            precisions.append(precision_score(self.y_true, y_pred, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred, zero_division=0))

            normal_mask = (self.y_true == 0)
            if normal_mask.sum() > 0:
                normal_errors = (y_pred[normal_mask] != 0).sum()
                normal_misrate.append(normal_errors / normal_mask.sum())
            else:
                normal_misrate.append(np.nan)

            anomaly_mask = (self.y_true == 1)
            if anomaly_mask.sum() > 0:
                anomaly_errors = (y_pred[anomaly_mask] != 1).sum()
                anomaly_misrate.append(anomaly_errors / anomaly_mask.sum())
            else:
                anomaly_misrate.append(np.nan)

        plt.figure(figsize=(10, 6))

        # Metriken
        plt.plot(thresholds, f1s, label="F1 Score", color="navy")
        plt.plot(thresholds, precisions, label="Precision", linestyle="-.", color="purple")
        plt.plot(thresholds, recalls, label="Recall", linestyle="-.", color="deeppink")

        # Fehlklassifikationsraten
        plt.plot(thresholds, normal_misrate, label="Fehler bei normalen (FP)", color="darkgreen", linestyle='--')
        plt.fill_between(thresholds, 0, normal_misrate, color="darkgreen", alpha=0.2)
        plt.plot(thresholds, anomaly_misrate, label="Fehler bei anomalen (FN)", color="red", linestyle='--')
        plt.fill_between(thresholds, 0, anomaly_misrate, color="red", alpha=0.2)

        # Threshold-Linie
        plt.axvline(self.threshold, color='gray', linestyle='--', label="Aktueller Threshold")

        plt.xlabel("Threshold")
        plt.ylabel("Wert (Metrik / Fehleranteil)")
        plt.title("Metriken & Fehlklassifikationsrate über Threshold")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        self._save_or_show(filename)



    def get_best_threshold(self):
        thresholds = np.linspace(0, 1, 500)
        f1s, precisions, recalls = [], [], []

        for t in tqdm(thresholds):
            y_pred = (self.y_score >= t).astype(int)
            f1s.append(f1_score(self.y_true, y_pred, zero_division=0))
            precisions.append(precision_score(self.y_true, y_pred, zero_division=0))
            recalls.append(recall_score(self.y_true, y_pred, zero_division=0))

        # Max F1 und zugehöriger Index
        idx_f1 = np.argmax(f1s)

        print(f"Max F1 Score:     {f1s[idx_f1]:.4f} bei Threshold = {thresholds[idx_f1]:.3f}")
        print(f"Precision bei max F1: {precisions[idx_f1]:.4f}")
        print(f"Recall bei max F1:    {recalls[idx_f1]:.4f}")

        return {
                "max_f1_score": float(f1s[idx_f1]),
                "threshold": float(thresholds[idx_f1]),
                "precision": float(precisions[idx_f1]),
                "recall": float(recalls[idx_f1]),
            } 

    def _save_or_show(self, filename):
        if self.save_path:
            full_path = os.path.join(self.save_path, filename)
            plt.savefig(full_path, dpi=300)
            plt.close()
        else:
            plt.show()


def compute(category, sourceDir, resultsDir):
    # JSON laden
    resultFile = f"{sourceDir}/{category}_test_results.json"
    with open(resultFile) as f:
        data = json.load(f)

    results = data["test_results"][category]
    y_true = [r["anomaly"] for r in results]
    y_score = [r["pr_sp"] for r in results]
    specie_true = [r["specie_name"] for r in results]

    # Evaluator erzeugen
    save_path = f"{resultsDir}/{category}"
    evaluator = AnomalyEvaluator(y_true, y_score, 
                                 threshold=0.5,
                                 specie_true=specie_true,
                                 save_path=save_path,
                                 )

    # Einzelne Methoden aufrufen
    bestMetrics = evaluator.get_best_threshold()
    evaluator.threshold = bestMetrics['threshold']
    evaluator.calculate_yPred()

    evaluator.print_metrics()
    evaluator.plot_precision_recall_curve()
    evaluator.plot_roc_curve()
    evaluator.plot_confusion_matrix_custom()
    evaluator.plot_confusion_matrix()
    evaluator.plot_score_distribution()
    evaluator.plot_threshold_vs_metrics()
    evaluator.plot_misclassification_by_threshold()
    evaluator.plot_all_metrics_combined()

    # Speichern falls gewünscht
    metricsJSON = f"{sourceDir}/{category}_metric_results.json"
    with open(metricsJSON) as f_out:
        metrics = json.load(f_out)
    metrics['metric_results'][category].update(bestMetrics)
    with open(metricsJSON, "w") as f_out:
        json.dump(metrics, f_out, indent=4)


if "__main__" == __name__:
    categorys = ['bottle', 'cable', 'grid', 'metal_nut', 'screw', 'wood']
    for category in categorys:
        try:
            compute(category,
                    sourceDir="/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_padim",
                    resultsDir="/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_padim/graphs/"
                    )
        except Exception as e:
            print(f"could not compute {category}")
            print(e)



