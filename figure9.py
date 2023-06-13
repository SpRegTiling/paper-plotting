import sys
import os
import matplotlib.ticker as mtick

sys.path.append(os.path.dirname(__file__))

from utils import *
from plot_utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams["figure.figsize"] = (6, 4.3)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

df = pd.read_csv(RESULTS_DIR + f"/figure9.csv")

fig, ax = plt.subplots()
sc1 = plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["required_storage_pct"], alpha=0.5, color='deepskyblue', s=1, label="Sparse Register Tiling")
sc2 = plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["csr_required_storage_pct"], alpha=0.5, color='firebrick', s=1, label='CSR')
lgnd = plt.legend(handles=[sc1, sc2], loc='upper right')
lgnd.legend_handles[0]._sizes = [50]
lgnd.legend_handles[1]._sizes = [50]
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

plt.ylabel('Required Storage (pct of dense)')
plt.xlabel('Sparsity')
plt.margins(x=0)
plt.tight_layout()
savefig("figure9.pdf")
