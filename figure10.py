import sys
import os
import pandas as pd
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter, NullFormatter
from numpy import mean

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (3, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def speedup_column(method, baseline):
    return f'Speed-up {method} vs. {baseline}'


def compute_speedup(df, method, baseline):
    df[speedup_column(method, baseline)] = df[f"time median|{baseline}"] / df[f"time median|{method}"]


MARKSIZE = 2.5
MARKSIZE_DL = 3

colors = {
    32: 'DarkBlue',
    128: 'DarkRed'
}

THREAD_COUNT = 20
BCOLS = 128

SS_MARKER = ">"
DL_MARKER = "."

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE - 1})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

YLAB = -0.48

fig, axs = plt.subplots(1, 4, figsize=(16, 6.1), squeeze=False, width_ratios=[1.5, 2, 1.1, 2])

##
#   All DLMC
##

ax = axs[0, 0]

all_dlmc_df = pd.read_csv(RESULTS_DIR + f"/figure10_all_dlmc.csv")
df = all_dlmc_df

handles = []
labels = []

chipset = "cascade"
x_labels = ['60%-69.9%', '70%-79.9%', '80%-89.9%', '90%-95%']
x_ticks = [i + 1 for i in range(len(x_labels))]


def _mean(list_in):
    geo_mean = []
    for sub_list in list_in:
        geo_mean.append(mean(sub_list))

    return geo_mean


mcl = intel_mcl_double
limits = (0, 500)

box_width = 0.15


def plot(ax, color, bias, data, label='nn'):
    geo_mean = _mean(data)
    ax.plot([x + bias - box_width * len(mcl) / 2 for x in x_ticks], geo_mean, color=color, linewidth=1)
    return ax.boxplot(data, positions=[x + bias - box_width * len(mcl) / 2 for x in x_ticks],
                      notch=True, patch_artist=True,
                      boxprops=dict(facecolor=color),
                      capprops=dict(color=color),
                      whiskerprops=dict(color=color),
                      flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=0.5),
                      medianprops=dict(color='black'),
                      showfliers=True,
                      # whis=(10,90),
                      widths=0.15)


# for method, _, _ in mcl:
#     df[f'Speed-up {method} vs. {baseline}'] = df[f"time cpu median|{baseline}"] / df[f"time cpu median|{method}"]

# axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
plots = []
labels = []

for mi, (method, color, label) in enumerate(mcl):
    data = [df[(df['sparsity'] >= spBucket * 0.1 + 0.6) & (df['sparsity'] < (spBucket + 1) * 0.1 + 0.6)
               & (df[f'correct|{method}'] == "correct")
               & (~df[f'gflops/s|{method}'].isna())
               ][f'gflops/s|{method}'].tolist() for spBucket in range(len(x_labels))]
    plots.append(plot(ax, color, box_width * mi, data))
    labels.append(label)

handles = [plot["boxes"][0] for plot in plots]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=22)
ax.set_xlim([0.5, len(x_labels) + 0.5])
ax.set_xlabel('Sparsity')
ax.set_ylabel(f'Required GFLOP/s')
ax.set_title(f'All DLMC (60%-95%)', pad=18)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylim(limits)
ax.text(0.45, YLAB, "(a)", transform=ax.transAxes, size=18)

fig.legend(handles, labels, loc='upper center', ncol=len(handles))

##
#   500 of Each
##

df = pd.read_csv(RESULTS_DIR + f"/figure10_combined_suitesparse_and_dlmc.csv")

#
#   Plot
#

mcl = mcl[2:]

ax = axs[0, 1]
df["sparsity_raw"] = df["sparsity_raw"]
df["density"] = 1 - df["sparsity_raw"]

ss = df["ss"]
df.loc[~ss & (df[
                  "density"] < 0.05), "density"] = 0.05  # normally we round so visually correct so minor outliers on the vertical line
ax = df[~ss].plot(kind='scatter', x="density", y='gflops/s|Sp. Reg.', c=SP_REG_COLOR, ax=ax, s=MARKSIZE_DL,
                  marker=DL_MARKER)
ax = df[ss].plot(kind='scatter', x="density", y='gflops/s|Sp. Reg.', c=SP_REG_COLOR, ax=ax, s=MARKSIZE,
                 marker=SS_MARKER)
ax = df[~ss].plot(kind='scatter', x="density", y='gflops/s|PSC', c=PSC_COLOR, ax=ax, s=MARKSIZE_DL, marker=DL_MARKER,
                  label='DLMC')
ax = df[ss].plot(kind='scatter', x="density", y='gflops/s|PSC', c=PSC_COLOR, ax=ax, s=MARKSIZE, marker=SS_MARKER,
                 label='SuiteSparse')
ax.set(xscale="log", yscale="linear")
ax.set_xlabel('Density (Log)')
ax.set_ylabel(None)
# ax.set_title(f'500 DLMC (60%-95%) & 500 SuiteSparse', pad=10)
ax.axvline(x=0.05, color='firebrick', linewidth=0.5)
ax.spines[['right', 'top']].set_visible(False)
ax.text(0.48, YLAB, "(b)", transform=ax.transAxes, size=18)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
plt.gca().xaxis.set_minor_formatter(NullFormatter())
legend = ax.legend(prop={'size': 12})

for handle in legend.legend_handles:
    handle._sizes = [30]
    handle.set_color('black')

ax = axs[0, 2]
dff = df[df["cov|Sp. Reg."] >= 0.01]


def plot(ax, color, bias, data, label='nn'):
    geo_mean = _mean(data)
    ax.plot([x + bias - box_width * len(mcl) / 2 for x in x_ticks], geo_mean, color=color, linewidth=1)
    return ax.boxplot(data, positions=[x + bias - box_width * len(mcl) / 2 for x in x_ticks],
                      notch=True, patch_artist=True,
                      boxprops=dict(facecolor=color),
                      capprops=dict(color=color),
                      whiskerprops=dict(color=color),
                      flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=0.5),
                      medianprops=dict(color='black'),
                      showfliers=True,
                      # whis=(10,90),
                      widths=0.15)


x_labels = ['0.01-0.1', '0.1-1', '1-10']
x_ranges = [(0.01, 0.1), (0.1, 1.0), (1, 10)]
x_ticks = [i + 1 for i in range(len(x_labels))]

for i in range(len(x_labels)):
    data = df[(df['cov|Sp. Reg.'] >= x_ranges[i][0]) & (df['cov|Sp. Reg.'] < x_ranges[i][1])]
    ss = data["ss"]

for mi, (method, color, label) in enumerate(mcl):
    data = [df[(df['cov|Sp. Reg.'] >= x_ranges[i][0]) & (df['cov|Sp. Reg.'] < x_ranges[i][1])
               & (df[f'correct|{method}'] == "correct")
               & (~df[f'gflops/s|{method}'].isna())
               ][f'gflops/s|{method}'].tolist() for i in range(len(x_labels))]
    plots.append(plot(ax, color, box_width * mi, data))
    labels.append(label)

handles = [plot["boxes"][0] for plot in plots]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=22)
ax.set_xlim([0.5, len(x_labels) + 0.5])
ax.set_xlabel('CoV Working Set Size')
ax.set_title(f'500 DLMC (60%-95%) & 500 SuiteSparse', pad=18)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylim(limits)
ax.text(0.46, YLAB, "(c)", transform=ax.transAxes, size=18)

cmap_reversed = matplotlib.colormaps.get_cmap('cividis_r')
mcl = mcl[2:]

ax = axs[0, 3]
df["sparsity_raw"] = df["sparsity_raw"]
df["density"] = 1 - df["sparsity_raw"]

ss = df["ss"]
df.loc[~ss & (df[
                  "density"] < 0.05), "density"] = 0.05  # normally we round so visually correct so minor outliers on the vertical line
s = ax.scatter(x=df[~ss]["avg_nnz_per_enumb|Sp. Reg."], y=df[~ss]["cov|Sp. Reg."], c=df[~ss]["gflops/s|Sp. Reg."],
               s=MARKSIZE_DL, marker=DL_MARKER, cmap=cmap_reversed, vmin=0, vmax=500, label="DLMC")
ax = df[ss].plot(kind='scatter', x="avg_nnz_per_enumb|Sp. Reg.", y='cov|Sp. Reg.', c='gflops/s|Sp. Reg.', ax=ax,
                 s=MARKSIZE, marker=SS_MARKER, colormap=cmap_reversed, vmin=0, vmax=500, label="SuiteSparse",
                 colorbar=False)


def xlabel(txt, l):
    return r'\begin{center}' + r'\\*\textit{\small{' + l + r'}}\end{center}'


ax.set(xscale="linear", yscale="log")
ax.set_xlabel('Avg NNZ per Enum. Block ($T_i$ = 4)')
ax.set_ylabel('CoV Working Set Size')
ax.spines[['right', 'top']].set_visible(False)
ax.text(0.48, YLAB, "(d)", transform=ax.transAxes, size=18)


cb = plt.colorbar(s)
cb.ax.set_ylabel('Required GFLOP/s (Sp. Reg. Tiling)')

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
plt.gca().xaxis.set_minor_formatter(NullFormatter())

legend_elements = [Line2D([0], [0], marker=DL_MARKER, color='black', label='DLMC',
                          linestyle='None',
                          markerfacecolor='black', markersize=5),
                   Line2D([0], [0], marker=SS_MARKER, color='black', label='SuiteSparse',
                          linestyle='None',
                          markerfacecolor='black', markersize=5)]

legend = ax.legend(handles=legend_elements, prop={'size': 12})

##
#   SAVE
##

plt.gcf().align_xlabels(axs[0, :])
plt.subplots_adjust(hspace=0.4, wspace=0.2)  # For cascadelake
plt.margins(x=0)
plt.tight_layout(rect=(0, 0, 1, 0.93))  # For cascadelake

savefig(f"/figure10.pdf")
