import sys
import os
import pandas as pd
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mtick

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(1, 2, figsize=(8,6))
xlim = (0,3)
ylim = (0.5,5)

df = pd.read_csv(RESULTS_DIR + f"/figure15_cascade.csv")

ax = axs[0]
ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1, ax=ax, colorbar=False)
ax.axhline(y=1.0, color='r', linestyle='-', linewidth=0.5)
ax.set_title('Versus MKL SpMM (CSR)', fontsize=FONT_SIZE+1)
ax.set_ylabel('Speedup')
ax.set_xlabel('Loads-per-FMA')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_xlim(xlim)

df = pd.read_csv(RESULTS_DIR + f"/figure15_raspberrypi.csv")

ax = axs[1]
ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1, ax=ax, colorbar=False)
ax.axhline(y=1.0, color='r', linestyle='-', linewidth=0.5)
ax.set_title('Versus XNN SpMM', fontsize=FONT_SIZE+1)
ax.set_ylabel(None)
ax.set_xlabel('Loads-per-FMA')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_xlim(xlim)

cmap = plt.get_cmap("cividis")
norm = plt.Normalize(60, 95)
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, pad=0.15, location='bottom', shrink=0.45)
cbar.ax.set_title("Sparsity", position=(-0.18,5.5), pad=-4, y=0.3, fontsize=16)
cbar.ax.tick_params(axis='both', which='major', labelsize=14)
cbar.ax.tick_params(axis='both', which='minor', labelsize=8)
cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
cbar.ax.set_xticks([x + 5 for x in cbar.ax.get_xticks()[:-1]])

savefig("figure15.pdf")