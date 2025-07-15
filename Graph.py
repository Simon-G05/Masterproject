import matplotlib.pyplot as plt
import numpy as np
import os
from adjustText import adjust_text

def save_or_show(filename, save_path=None):
    if save_path:
        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path, dpi=300)
        plt.close()
    else:
        plt.show()
nImg = 665/1000
# Beispielhafte Datenpunkte: (Latency, AUROC, Label)
data = [
    (59 / nImg, 97.43, "EfficientAD"),
    (49.59 / nImg, 98.5, "Patchcore"),
    (56.31 / nImg, 88.49, "Padim"),
    (758.12/ nImg, 89.2, "AnomalyCLIP\n Zero Shot"),
    (750 /nImg , 93.1, "AnomalyCLIP\n Few Shot"),
]

# Extrahiere Daten
latencies = [d[0] for d in data]
aurocs = [d[1] for d in data]
labels = [d[2] for d in data]

# Plot vorbereiten
fig, ax = plt.subplots()
ax.set_xscale('log')  # log-Skala für x-Achse
ax.set_xlabel('Latency per image [ms]')
ax.set_ylabel('AU-ROC [%]')
ax.set_title('AU-ROC vs. Latency')

# Punkte plotten
ax.plot(latencies, aurocs, 'rx')  # rotes Kreuz

# Beschriftungen hinzufügen
texts = []
for x, y, label in zip(latencies, aurocs, labels):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), ha='left')
    # texts.append(ax.text(x, y, label, fontsize=9))

#adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(80,100)
plt.xlim(10,10000)
plt.tight_layout()

save_or_show('aurocLatency.png', '/Users/simon/Documents/HS_Kempten/Projekt/Git/Masterprojekt/Programme/Visualization')
