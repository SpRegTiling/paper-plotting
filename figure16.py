import sys
import os
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(__file__))

from utils import *

df = pd.read_csv(RESULTS_DIR + f"/figure16.csv")

color_labels = df['search'].unique()
rgb_values = sns.color_palette("Set2", 5)
color_map = dict(zip(color_labels, rgb_values))

ax = df.plot.scatter(x='num_patterns', y='gflops/s', c=df["search"].map(color_map), alpha=0.8, s=10)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

markers = []
for name, color in color_map.items():
    markers.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=name))

plt.xlim(0, 259)
plt.ylabel('Required GFLOP/s')
plt.xlabel('Number of Generated Enumerated Blocks')
plt.legend(handles=markers)
savefig("figure16.pdf")
