import sys
import os
import pandas as pd
from brokenaxes import brokenaxes

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

df = pd.read_csv(RESULTS_DIR + f"/figure12.csv")

plt.figure(figsize=(4, 5.5))
bax = brokenaxes(xlims=((0, 2250), (33200, 33400)), hspace=.02)


def plot(model, color):
    dff = df[df["model"] == model]
    dfg = dff.groupby(['Rows', 'Cols']).size().reset_index().rename(columns={0: 'count'})
    bax.scatter('Rows', 'Cols',
                s='count',
                c=color,
                alpha=0.4,
                data=dfg)

plot("transformer", "navy")
plot("rn50", "maroon")

for ax in bax.axs:
    ax.set_ylim(0, 5100)
bax.standardize_ticks(512, 512)
bax.set_xlabel("Rows", labelpad=28)
bax.set_ylabel("Columns", labelpad=40)
legend = bax.legend(labels=["Transformer", "ResNet50"])

for handle in legend.legend_handles:
    handle._sizes = [30]

savefig(f"/figure12.pdf")

