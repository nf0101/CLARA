"""Dot plot con barre di errore per la griglia cap x time_decay."""
from pathlib import Path
import matplotlib.pyplot as plt

# (cap, decay, media, std) - Tabella 15
DATA = [
    (0.3, 0.85, 0.0899, 0.0102),
    (0.3, 0.95, 0.1001, 0.0024),
    (0.3, 1.00, 0.0968, 0.0123),
    (0.5, 0.85, 0.0938, 0.0086),
    (0.5, 0.95, 0.1030, 0.0013),
    (0.5, 1.00, 0.0967, 0.0077),
    (0.7, 0.85, 0.0967, 0.0040),
    (0.7, 0.95, 0.0989, 0.0031),
    (0.7, 1.00, 0.0936, 0.0036),
]
DEFAULT = (0.5, 1.00)
FEDAVG, FEDAVG_SD = 0.0911, 0.0021

ACCENT, MUTED, INK, GRID = "#2f6fb5", "#9aa5b1", "#1f2933", "#dde3e8"

rows = sorted(DATA, key=lambda r: r[2])
grid_mean = sum(r[2] for r in DATA) / len(DATA)

fig, ax = plt.subplots(figsize=(8, 5.2))

ax.axvspan(FEDAVG - FEDAVG_SD, FEDAVG + FEDAVG_SD, color=MUTED, alpha=0.18, lw=0)
ax.axvline(FEDAVG, color=MUTED, lw=1.4, ls="--", zorder=1)
ax.axvline(grid_mean, color=ACCENT, lw=1.2, ls=":", alpha=0.8, zorder=1)

for y, (cap, decay, mean, sd) in enumerate(rows):
    is_def = (cap, decay) == DEFAULT
    c = ACCENT if is_def else MUTED
    ax.errorbar(mean, y, xerr=sd, fmt="o", color=c, ecolor=c,
                elinewidth=1.6, capsize=4, capthick=1.6,
                markersize=11 if is_def else 8,
                markeredgecolor="white", markeredgewidth=1.6, zorder=3)

ax.set_yticks(range(len(rows)))
ax.set_yticklabels(
    [f"cap {c:.1f}  ·  decay {d:.2f}" for c, d, _, _ in rows],
    fontsize=10.5)
for lbl, (cap, decay, _, _) in zip(ax.get_yticklabels(), rows):
    if (cap, decay) == DEFAULT:
        lbl.set_color(ACCENT)
        lbl.set_fontweight("bold")

ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_xlabel("critical_ap50 al round 10 (media su tre seed, barre $\\pm 1\\sigma$)",
              fontsize=10.5, color=INK)
ax.set_title("Ablation: nessuna configurazione domina entro il rumore",
             fontsize=12.5, color=INK, pad=14)

ax.annotate("FedAvg", xy=(FEDAVG, len(rows) - 0.55), xytext=(-6, 0),
            textcoords="offset points", ha="right", va="center",
            fontsize=9.5, color="#6b7682")
ax.annotate("media griglia", xy=(grid_mean, -0.45), xytext=(5, 0),
            textcoords="offset points", ha="left", va="center",
            fontsize=9.5, color=ACCENT)
ax.annotate("configurazione usata\nin RQ1 e RQ2",
            xy=(0.0967 + 0.0077, rows.index((0.5, 1.00, 0.0967, 0.0077))),
            xytext=(12, 14), textcoords="offset points",
            fontsize=9.5, color=ACCENT, linespacing=1.35,
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1))

ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.yaxis.grid(False)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="both", length=0, colors=INK, labelsize=10)

plt.tight_layout()
out = Path("out_dot/ablation_dotplot.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("ok", out)