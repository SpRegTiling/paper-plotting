import os
import sys
import pandas as pd
import matplotlib.ticker as mtick

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE + 2})
plt.rcParams["figure.figsize"] = (16, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


PLOT_GFLOPS = True
PLOT_SPEEDUP = True
ADD_REF_LINE = False
color_scheme = 'purpleblue'

methods = {
    'Best Nano': 'Best Nano-Kernel',
    'ASpT': 'ASpT',
    'MKL_Dense': 'MKL Dense',
    'MKL_Sparse': 'MKL Sparse'
}

df = pd.read_csv(RESULTS_DIR + f"/figure14.csv")

density_thresh = 1.0
density_thresh_2 = 0.3
alpha = 0.9

bColsList = [256]
fig, axs = plt.subplots(1, 3)
handles, labels = [], []
for n in range(len(bColsList)):
    df_filtered = df[(df["numThreads"] == 1) \
                     & (df["density"] <= density_thresh) \
                     & (df["n"] == bColsList[n])]

    df_filtered = df_filtered.sort_values(by=['sparsity'])

    idx = 0
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity'] * 100,
                df_filtered[df_filtered['name'] == method]['time median'] / 1e3, alpha=alpha, c=color, label=label)
    ax.set_ylabel(f'Execution Time (ms)')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if n == 0:
        handles.extend(ax.get_legend_handles_labels()[0])
        labels.extend(ax.get_legend_handles_labels()[1])

    idx += 1
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity'] * 100,
                df_filtered[df_filtered['name'] == method]['SP_FLOPS_TOTAL-mkl'] / 1e6, alpha=alpha, c=color,
                label=label)
    ax.set_ylabel(f'Redundant MFLOPs')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    idx += 1
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity'] * 100,
                df_filtered[df_filtered['name'] == method]['loads_per_fma'], alpha=alpha, c=color, label=label)
    ax.set_ylabel(f'Loads-per-FMA')
    ax.set_ylim(0, 2.8)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

for ax in axs:
    ax.axvline(x=95, color='black', linewidth=0.5, linestyle=(0, (3, 5)), alpha=0.7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Sparsity')
    lim = ax.get_xlim()
    ax.set_xticks(list(ax.get_xticks()) + [95])
    ax.set_xlim(lim)

plt.subplots_adjust(hspace=0.3, wspace=0.3)
fig.legend(handles, labels, loc='upper center', ncol=len(handles))

plt.margins(x=0)
plt.tight_layout(rect=(0, 0, 1, 0.92))
savefig('figure14.pdf')