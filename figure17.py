import json
import sys
import numpy as np
import os
from collections import defaultdict

sys.path.append(os.path.dirname(__file__))

from utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': FONT_SIZE - 1})
plt.rcParams["figure.figsize"] = (16, 4.7)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.figure(figsize=(16, 4))


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


BENCHMARK_NAME_MAP = {
    "FP32MobileNetV1": ("XNN SGEMM", 0.0),
    "Empty": ("_hid", -1),
    "FP32Sparse70MobileNetV1": ("XNN SpMM", 0.7),
    "FP32Sparse80MobileNetV1": ("XNN SpMM", 0.8),
    "FP32Sparse90MobileNetV1": ("XNN SpMM", 0.9),
    "FP32Sparse70MobileNetV1Nano": ("Sp. Reg. Tiling", 0.7),
    "FP32Sparse80MobileNetV1Nano": ("Sp. Reg. Tiling", 0.8),
    "FP32Sparse90MobileNetV1Nano": ("Sp. Reg. Tiling", 0.9),
}

results = json.load(open(RESULTS_DIR + '/end2end_bench_v1.json'))
optimized_layers = list(range(3, 28, 2))

results_per_threadcount = defaultdict(lambda: {})
dense_baseline_times_per_threadcount = {}
sparse_baseline_times_per_threadcount = defaultdict(lambda: {})
total_time_per_threadcount = defaultdict(lambda: {})


def Label(x):
    if x[1] < 0: return ""
    if x[1] == 0:
        return "XNN SGEMM"
    if "XNN" in x[0]:
        return f'{x[0]:<12} ({round(100 * x[1])}%)'
    else:
        return f'{x[0]:<16} ({round(100 * x[1])}%)'


for benchmark in results["benchmarks"]:
    benchmark_name = BENCHMARK_NAME_MAP[benchmark["name"].split("/")[0]]
    threads = int(benchmark["name"].split("/")[1].split(":")[1])

    layer_times = []
    for key, value in benchmark.items():
        if "layer" in key:
            layer = int(key.split("_")[1])
            layer_times.append((layer, value))

    layer_times = sorted(layer_times, key=lambda x: x[0])
    layer_times = np.array([x[1] for x in layer_times])

    results_per_threadcount[threads][benchmark_name] = layer_times
    total_time_per_threadcount[threads][benchmark_name] = benchmark["real_time"]

    if benchmark_name[1] == 0.0:
        dense_baseline_times_per_threadcount[threads] = benchmark["real_time"]

    if benchmark_name[0] == "XNN SpMM":
        sparse_baseline_times_per_threadcount[threads][benchmark_name[1]] = benchmark["real_time"]


threads = 4
fig, axs = plt.subplots(figsize=(10, 3.1))

LINE_STYLES = {
    "XNN SGEMM": '-',
    "XNN SpMM": ':',
    "Sp. Reg. Tiling": '--',
}

COLORS = {
    0.0: 'black',
    0.7: 'red',
    0.8: 'blue',
    0.9: 'green',
}

def plot(bbenchmark_name, times):
    line_style = LINE_STYLES[benchmark_name[0]]
    color = COLORS[benchmark_name[1]]
    axs.plot(range(1, len(times) + 1), np.cumsum(times) / 1000, marker='x', label=Label(benchmark_name),
             linestyle=line_style, color=color, linewidth=1.5)


benchmark_name, times = list(results_per_threadcount[threads].items())[0]
plot(benchmark_name, times)
plot(("", -1), [])

for benchmark_name, times in list(results_per_threadcount[threads].items())[1:]:
    plot(benchmark_name, times)

for i in optimized_layers:
    # only one line may be specified; full height
    axs.axvline(x=i, color='b', label=None, linewidth=0.5)

leg = axs.legend(loc='upper left', ncols=4, fontsize=11, columnspacing=0.8, shadow=True)
leg.legend_handles[1]._visible = False
leg.texts[1]._visible = False
axs.set_ylabel('Cumulative Time (ms)')
axs.set_xlabel('Layer')
axs.spines.right.set_visible(False)
axs.spines.top.set_visible(False)

plt.margins(x=0)
plt.tight_layout()
savefig(f'figure17.pdf')