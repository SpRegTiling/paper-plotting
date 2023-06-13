import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 25})
plt.rcParams["figure.figsize"] = (12.5, 8)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

df = pd.read_csv(RESULTS_DIR + f"/figure2.csv")

"""
Model name:            Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
Stepping:              7
CPU MHz:               3200.073
CPU max MHz:           3900.0000
CPU min MHz:           1000.0000
"""

freq = 3_200_000_000
compute_bound_min_time = (df["SP_FLOPS_TOTAL"].min() / (freq*2*16)) * 1_000_000
load_bound_min_time = ((df["SP_FLOPS_TOTAL"].min() / (freq*2*16)) *2.5) * 1_000_000

fig, ax = plt.subplots()
color_scheme = 'Set2'
df.sort_values(by=['SP_FLOPS_TOTAL'], inplace=True)

codes = [i for i in range(len(df))]
ax.plot([0, 1], [compute_bound_min_time, compute_bound_min_time], color="blue", linestyle=":", label="Compute Bound")
ax.plot([1, 2.5], [compute_bound_min_time, load_bound_min_time], color="blue", linestyle=":", label="Load Bound")
plt.axvline(x=1, color="black", linestyle="--", alpha=0.25)
sc = ax.scatter(df['loads_per_fma'], df['time median'], alpha=1, s=df['SP_FLOPS_TOTAL']/df['SP_FLOPS_TOTAL'].max() * 5000, cmap=color_scheme, c=codes, label="Sparse Register Tiling")

ax.annotate(f"{df['name'].tolist()[0]}\n({int(df['SP_FLOPS_TOTAL'].tolist()[0] / 1e6)})", (df['loads_per_fma'].tolist()[0], df['time median'].tolist()[0]),\
    xytext=(df['loads_per_fma'].tolist()[0] + 0.08, df['time median'].tolist()[0] - 250), textcoords='data')

ax.annotate(f"{df['name'].tolist()[1]}\n({int(df['SP_FLOPS_TOTAL'].tolist()[1] / 1e6)})", (df['loads_per_fma'].tolist()[1], df['time median'].tolist()[1]),\
    xytext=(df['loads_per_fma'].tolist()[1] + 0.09, df['time median'].tolist()[1] - 250), textcoords='data')

ax.annotate(f"{df['name'].tolist()[2]}\n({int(df['SP_FLOPS_TOTAL'].tolist()[2] / 1e6)})", (df['loads_per_fma'].tolist()[2], df['time median'].tolist()[2]),\
    xytext=(df['loads_per_fma'].tolist()[2] - 0.09, df['time median'].tolist()[2] - 250), textcoords='data', horizontalalignment='right')

ax.annotate(f"{df['name'].tolist()[3]}\n({int(df['SP_FLOPS_TOTAL'].tolist()[3] / 1e6)})", (df['loads_per_fma'].tolist()[3], df['time median'].tolist()[3]),\
    xytext=(df['loads_per_fma'].tolist()[3] - 0.15, df['time median'].tolist()[3] - 250), textcoords='data', horizontalalignment='right')

ax.annotate(f"{df['name'].tolist()[4]}\n({int(df['SP_FLOPS_TOTAL'].tolist()[4] / 1e6)})", (df['loads_per_fma'].tolist()[4], df['time median'].tolist()[4]),\
    xytext=(df['loads_per_fma'].tolist()[4] + 0.14, df['time median'].tolist()[4] - 250), textcoords='data')

ax.annotate(f"Theoretical Limit", (1.5, (load_bound_min_time + compute_bound_min_time) / 3),\
    xytext=(1.65, (load_bound_min_time + compute_bound_min_time) / 3), textcoords='data')

ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylim([0, 5500])
plt.xlabel("Loads-per-FMA")
plt.ylabel("Execution Time (\u00B5s)")
plt.tight_layout()
savefig("figure2.pdf")
