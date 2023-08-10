import os.path
import sys
from pathlib import Path

import pandas

sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER, POWER_MODE
import matplotlib.ticker as mticker


baseE = {
    "0": {
        'faster-agx': 8639.9460,
        'yolor-agx': 7679.2450,
        'yolox-agx': 7179.3893,
        'yolov4-agx': 4213.3583,
        'faster-nano': 14213.7446,
        'yolor-nano': 10447.9559,
        'yolox-nano': 13460.1792,
        'yolov4-nano': 9055.3908,
        'yolos-agx': 95233.7091,
    },
    "1": {
        'faster-agx': 8724.0558,
        'yolor-agx': 9049.7275,
        'yolox-agx': 24857.6348,
        'yolov4-agx': 7468.0064,
        'faster-nano': 12406.4492,
        'yolor-nano': 9158.5191,
        'yolox-nano': 38119.5639,
        'yolov4-nano': 10734.7091,
        'yolos-agx': 95233.7091,
    }

}

baseBattery = {
    'faster-agx': 1,
    'yolor-agx': 1,
    'yolox-agx': 1,
    'yolov4-agx': 1,
    'faster-nano': 1,
    'yolor-nano': 1,
    'yolox-nano': 1,
    'yolov4-nano': 1,
    'yolos-agx': 1,
}

swap_mem_baseline = {
    "2GB": {
        "yolos-agx": 12321,
        "yolox-agx": 5091.6,
        "yolor-agx": 3250,
        "yolov4-agx": 3547,
        "faster-agx": 3055.5,
        "yolox-nano": 5091.6,
        "yolor-nano": 3250,
        "yolov4-nano": 3547,
        "faster-nano": 3055.5,
    },
    "4GB": {
        "yolos-agx": 10504,
        "yolox-agx": 2800,
        "yolor-agx": 1206,
        "yolov4-agx": 1607,
        "faster-agx": 715,
        "yolox-nano": 2800,
        "yolor-nano": 1206,
        "yolov4-nano": 1607,
        "faster-nano": 715,
    }
}

benchmarks = {
        "0": {
            'faster-agx': 0.349021,
            'faster-nano': 1.967904,
            'faster-clarity32': 0.063555,
            'yolor-agx': 0.1736289,
            'yolor-nano': 1.458861,
            'yolox-agx': 2.6009,
            'yolox-nano': 1.6617,
            'yolov4-agx': 0.1969,
            'yolov4-nano': 1.1422,
            'yolos-agx': 6.086,
        },
        "1": {
            'faster-agx': 1.157999,
            'faster-nano': 2.686923,
            'faster-clarity32': 0.063555,
            'yolor-agx': 0.369261,
            'yolor-nano': 2.011526,
            'yolox-agx': 1.0978,
            'yolox-nano': 2.3129,
            'yolov4-agx': 0.6596,
            'yolov4-nano': 1.5723,
            'yolos-agx': 6.086,
        },
    }

swap_power = {
    'yolox-nano': 2561,
    'faster-nano': 2740,
    'yolor-nano': 2732,
    'yolov4-nano': 2566,
    'yolos-agx': 2854
}

break_points = {
    'yolox-nano': [500, 3000],
    'faster-nano': [300, 1000],
    'yolor-nano': [100, 1000],
    'yolov4-nano': [125, 1100],
    'yolos-agx': [5000, 19000]
}

avg_battery = []

def draw(config):
    power = "1"
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # ax = fig.add_subplot(111)  # The big subplot
    fig.subplots_adjust(hspace=0.5)  # adjust space between axes
    fig.set_size_inches(4,2)

    df_2GB = pd.read_csv(f"../data/final_data/mem_constrained/2GB/{config}.csv")
    if len(df_2GB['bandwidth']) == 0:
        print("No results printed")
        return

    x1_list = []
    for i in df_2GB['bandwidth']:
        x1_list.append(i)

    base = 0
    if baseE[power][config] != 0:
        base = baseE[power][config]
    df_2GB['energy'] += base
    plot_data = df_2GB['energy']/1000
    for i in range(len(plot_data)):
        plot_data[i] = plot_data[i] * ((1-df_2GB['optimizer'][i]*0.01) * benchmarks['1'][config])

    w = (df_2GB['bandwidth'][len(df_2GB['bandwidth']) - 1] - df_2GB['bandwidth'][0]) / len(df_2GB) * 0.55
    b1 = ax1.bar(x1_list, plot_data, width=w, label='SN', color=sns.xkcd_rgb["maize"])
    swap_2G_USB = (base + swap_power[config] * swap_mem_baseline['2GB'][config] / 1280) * (benchmarks['1'][config] + swap_mem_baseline['2GB'][config] / 1280)
    l1 = ax1.axhline(y=swap_2G_USB/1000, color='b', linestyle='--', label='Baseline (USB)')
    swap_2G_MicroSD = (base + swap_power[config] * swap_mem_baseline['2GB'][config] / 190) * (benchmarks['1'][config] + swap_mem_baseline['2GB'][config] / 190)
    l3 = ax1.axhline(y=swap_2G_MicroSD/1000, color='b', linestyle='-', label='Baseline (MicroSD)')

    b1 = ax2.bar(x1_list, plot_data, width=w, label='SN', color=sns.xkcd_rgb["maize"])
    swap_2G_USB = (base + swap_power[config] * swap_mem_baseline['2GB'][config] / 1280) * (benchmarks['1'][config] + swap_mem_baseline['2GB'][config] / 1280)
    l1 = ax2.axhline(y=swap_2G_USB/1000, color='b', linestyle='--', label='Baseline (USB)')
    swap_2G_MicroSD = (base + swap_power[config] * swap_mem_baseline['2GB'][config] / 190) * (benchmarks['1'][config] + swap_mem_baseline['2GB'][config] / 190)
    l3 = ax2.axhline(y=swap_2G_MicroSD/1000, color='b', linestyle='-', label='Baseline (MicroSD)')

    ax2.set_ylim(0, break_points[config][0])  # outliers only
    ax1.set_ylim(break_points[config][1], ax1.get_ylim()[1])  # most of the data

    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    fig.text(0.5, -0.1, 'Bandwidth (Mbps)', ha='center', fontsize=12)
    fig.text(-0.04, 0.5, 'EDP', va='center', rotation='vertical', fontsize=12)

    # Set legends
    plt.legend(handles=[l3, l1, b1], bbox_to_anchor=(1,1.4), loc="right", prop={'size': 8})
    # plt.grid()
    plt.savefig(f"EDP/{config}-2G.png", bbox_inches='tight', dpi=100)
    plt.savefig(f"EDP/pdfs/{config}-2G.pdf", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        # 'faster-agx',
        'faster-nano',
        # 'yolor-agx',
        'yolor-nano',
        # 'yolox-agx',
        'yolox-nano',
        # 'yolov4-agx',
        'yolov4-nano',
         'yolos-agx'
    ]

    for config in tqdm(configs):
        draw(config)

    # print(sum(avg_battery)/len(avg_battery))