import re
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

def _save_or_show(save_path):
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def readOutFile(logfile, resultpath):
    # Muster für Losses pro Batch
    loss_pattern = re.compile(
        r"Epoch\s*(\d+):.*?\|\s*(\d+)/(\d+).*?train_st_step=([\d\.]+),\s*train_ae_step=([\d\.]+),\s*train_stae_step=([\d\.]+),\s*train_loss_step=([\d\.]+)"
    )
    # Muster für Trainingszeit
    time_pattern = re.compile(
        r"Training Time:\s*([\d\.]+)\s*seconds,\s*([\d\.]+)\s*minutes"
    )

    class_path = re.compile(
        r"Model saved to /workspace/Master/results/train_ocm/(\w+)/(\w+)_(\d+_\d+).pth"
    )

    # Statusvariablen
    current_class = None
    timestamp = None
    time_match = None
    loss_records = []
    time_records = []

    with open(logfile, "r") as f:
        for line in f:
            # Losses extrahieren
            loss_match = loss_pattern.search(line)
            if loss_match:
                epoch = int(loss_match.group(1))
                batch = int(loss_match.group(2))
                total_batches = int(loss_match.group(3))
                train_st_step = float(loss_match.group(4))
                train_ae_step = float(loss_match.group(5))
                train_stae_step = float(loss_match.group(6))
                train_loss_step = float(loss_match.group(7))
                loss_records.append({
                    "epoch": epoch,
                    "batch": batch,
                    "total_batches": total_batches,
                    "train_st_step": train_st_step,
                    "train_ae_step": train_ae_step,
                    "train_stae_step": train_stae_step,
                    "train_loss_step": train_loss_step
                })

            class_match = class_path.search(line)
            if class_match:
                current_class = class_match.group(1)
                timestamp = class_match.group(3)

                # DataFrames erzeugen
                df_losses = pd.DataFrame(loss_records)
                
                df_losses['global_batch'] = (df_losses['epoch'] + 1) * df_losses['batch']

                # Abspeichern (optional)
                path = os.path.join(resultpath, f"losses_{current_class}.csv")
                df_losses.to_csv(path, index=False)
                
                print(current_class)
                print("Losses DataFrame:")
                print(df_losses.head())

                if time_match:
                    seconds = float(time_match.group(1))
                    minutes = float(time_match.group(2))
                    time_records.append({
                        "class": current_class,
                        "training_time_seconds": seconds,
                        "training_time_minutes": minutes,
                        "timestamp": timestamp
                    })
                    # Nach Trainingsende zurücksetzen, falls mehrere Klassen folgen
                    # current_class = None
                loss_records = []

            # Trainingszeit extrahieren
            time_match = time_pattern.search(line)

    df_times = pd.DataFrame(time_records)
    path = os.path.join(resultpath, "training_time.csv")
    df_times.to_csv(path, index=False)
    print("\nTrainingszeit DataFrame:")
    print(df_times.head())

def extractClassname(classname, pattern=r"losses_(\w+).csv"):
    classmatch = re.match(pattern, classname)
    if classmatch:
        classname = classmatch.group(1)
    return classname

def plotCSV(sourcepath, resultpath=None, graphs=['train_loss_step', 'train_st_step', 'train_ae_step', 'train_stae_step']):
    # Losses für eine Klasse (z.B. metal_nut) einlesen
    df_losses = pd.read_csv(sourcepath)
    # Nach (epoch, batch) gruppieren und Mittelwert berechnen
    df_losses = df_losses.groupby(["global_batch"], as_index=False).mean(numeric_only=True)

    psourcepath = Path(sourcepath)
    source_dirs = psourcepath.parts
    classname = extractClassname(source_dirs[-1])
    out_path = os.path.join(path_graphs, f"{classname}.png") if resultpath is not None else None

    plt.figure(figsize=(10, 6))
    for graph in graphs:
        try:
            plt.plot(df_losses['global_batch'], df_losses[graph], label=graph)
        except:
            print(f"could not plot graph: {graph}")

    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Loss-Kurven für {classname}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    _save_or_show(out_path)

def plotTimes(sourcepath, resultpath):
    # Trainingszeiten aller Klassen einlesen
    df_times = pd.read_csv(sourcepath)
    # --- Trainingszeiten für alle Klassen als Balkendiagramm ---
    plt.figure(figsize=(8, 5))
    plt.bar(df_times['class'], df_times['training_time_minutes'])
    plt.xlabel('Klasse')
    plt.ylabel('Trainingszeit (Minuten)')
    plt.title('Trainingszeiten pro Klasse')
    plt.tight_layout()
    out_path = os.path.join(path_graphs, "times.png") if resultpath is not None else None
    _save_or_show(out_path)

    total_Time = df_times['training_time_minutes'].sum()
    print(f"total Time: {total_Time} min ({total_Time / 60:.2f} h)")


if "__main__" == __name__:
    logfile = "/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_efficientAD2/oc_anomalib.5174021.lrz-hgx-h100-011.out"
    resultpath = "/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_efficientAD2"
    # readOutFile(logfile, resultpath)
    
    # Alle CSV-Dateien im Ordner auflisten
    csv_dateien = glob(os.path.join(resultpath, "losses_*.csv"))
    for csv in csv_dateien:
    # csv = "/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/results/test_ocm_efficientAD2/losses_screw.csv"
        path_graphs = os.path.join(resultpath, "graphs")
        os.makedirs(path_graphs, exist_ok=True)
        plotCSV(csv, path_graphs)
    times = os.path.join(resultpath, 'training_time.csv')
    plotTimes(times, path_graphs)