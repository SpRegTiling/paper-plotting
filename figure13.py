import sys
import os
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv(RESULTS_DIR + f"/figure13.csv")

bColsList = [32, 128, 256, 512]
numThreadsList = [1]
fig, axs = plt.subplots(len(numThreadsList), len(bColsList), figsize=(16, 4.3))
plt.locator_params(nbins=4)
dimw = 0.6
alpha = 1

adf = df[df['name'] == 'transformed']['gflops/s'] - df[df['name'] == 'not-transformed']['gflops/s']
adf = adf[adf.notna()]
a1 = (df[df['name'] == 'transformed']['gflops/s'] - df[df['name'] == 'not-transformed']['gflops/s']).mean()
a2 = (df[df['name'] == 'not-transformed']['gflops/s'] - df[df['name'] == 'dense']['gflops/s']).mean()

handles, labels = [], []
for numThreads in range(len(numThreadsList)):
    dftmp = filter(df, numThreads=numThreadsList[numThreads])
    np.random.seed(0)
    paths = np.random.choice(dftmp["matrixId"].unique(), 20, replace=False)
    merged_chart = None

    for bcols in range(len(bColsList)):
        df_filtered = filter(dftmp, matrixId=list(paths), n=bColsList[bcols])

        df_filtered = df_filtered.sort_values(by=['sparsity'])
        axs[bcols].text(6, 80, f"B columns = {bColsList[bcols]}", fontsize=15)

        x = np.arange(len(df_filtered[df_filtered['name'] == 'transformed']['sparsity']) - 2) + 1

        axs[bcols].bar(x, df_filtered[df_filtered['name'] == 'transformed']['gflops/s'][:-2], dimw, color='royalblue',
                       alpha=alpha, label='unroll-and-sparse-jam + data compression')
        axs[bcols].bar(x, df_filtered[df_filtered['name'] == 'not-transformed']['gflops/s'][:-2], dimw, color='salmon',
                       alpha=0.8, label='unroll-and-sparse-jam')
        axs[bcols].bar(x + dimw / 2, df_filtered[df_filtered['name'] == 'dense']['gflops/s'][:-2], dimw / 2,
                       color='green', alpha=alpha, label='MKL SGEMM')
        if bcols == 0 and numThreads == 0:
            handles.extend(axs[bcols].get_legend_handles_labels()[0])
            labels.extend(axs[bcols].get_legend_handles_labels()[1])
        axs[bcols].spines.right.set_visible(False)
        axs[bcols].spines.top.set_visible(False)
        axs[bcols].set_xticks(x)
        axs[bcols].set_xlabel('Matrix Instance')
        axs[bcols].set_ylim(0, 85)
        axs[bcols].yaxis.set_major_locator(ticker.MultipleLocator(20))

        every_nth = 3
        for n, label in enumerate(axs[bcols].xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

axs[0].set_ylabel('Required GFLOP/s')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

fig.legend(handles, labels, loc='upper center', framealpha=0.3, ncol=len(handles))
plt.margins(x=0)
plt.tight_layout(rect=(0, 0, 1, 0.92))
savefig("figure13.pdf")
