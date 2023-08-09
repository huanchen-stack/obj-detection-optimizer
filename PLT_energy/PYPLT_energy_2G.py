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

swap_power = {
    'yolox-nano': 2561,
    'faster-nano': 2740,
    'yolor-nano': 2732,
    'yolov4-nano': 2566,
}

avg_battery = []

def draw(config):
    power = "1"
    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 3)
    df_2GB = pd.read_csv(f"../data/final_data/mem_constrained/2GB/{config}.csv")
    # df_4GB = pd.read_csv(f"../data/final_data/mem_constrained/4GB/{config}.csv")
    if len(df_2GB['bandwidth']) == 0:
        print("No results printed")
        return

    x1_list = []
    for i in df_2GB['bandwidth']:
        x1_list.append(i)

    base = 0
    if baseE[power][config] != 0:
        base = baseE[power][config]
    # battery_plot_data = baseBattery[config] * df_2GB['device'] / (baseE[POWER_MODE][config] + df_2GB['energy']) / (
    #         baseBattery[config] / baseE[POWER_MODE][config])

    # avg_battery.append(battery_plot_data[len(battery_plot_data)-1])
    # avg_battery.append(battery_plot_data[0])

    w = (df_2GB['bandwidth'][len(df_2GB['bandwidth']) - 1] - df_2GB['bandwidth'][0]) / len(df_2GB) * 0.55
    b0 = ax1.bar(x1_list, base/1000, width=w, label='Energy: Computation',
                 color=sns.xkcd_rgb["denim blue"])
    b1 = ax1.bar(x1_list, df_2GB['energy']/1000, width=w, label='Energy: Communication, 2G',
                 bottom=base/1000, color=sns.xkcd_rgb["maize"])


    swap_2G_USB = base + swap_power[config]  * swap_mem_baseline['2GB'][config] / 1280
    l1 = ax1.axhline(y=swap_2G_USB/1000, color='b', linestyle='--', label='USB swap baseline')
    swap_2G_MicroSD = base + swap_power[config]  * swap_mem_baseline['2GB'][config] / 190
    l3 = ax1.axhline(y=swap_2G_MicroSD/1000, color='b', linestyle='-', label='MicroSD swap baseline')

    if config == 'faster-nano':
        ax1.set_ylim(ymax=60)


    # ax2 = ax1.twinx()
    # w = (df_4GB['bandwidth'][len(df_4GB['bandwidth']) - 1] - df_4GB['bandwidth'][0]) / len(df_4GB) * 0.25
    # b2 = ax1.bar(x1_list, base, width=w, label='Energy: Computation',
    #              color=sns.xkcd_rgb["denim blue"])
    # b3 = ax1.bar(x1_list, df_4GB['energy'], width=w, label='Energy: Communication, 4G',
    #              bottom=base, color=sns.xkcd_rgb["red"])
    # swap_4G_USB = base + 2800 * swap_mem_baseline['4GB'][config] / 1280
    # l0 = ax1.axhline(y=swap_4G_USB, color='r', linestyle='-', label='swap 4G USB')
    # swap_4G_MicroSD = base + 2800 * swap_mem_baseline['4GB'][config] / 190
    # l2 = ax1.axhline(y=swap_4G_MicroSD, color='b', linestyle='-', label='swap 4G MicroSD')

    ######################################## Battery life
    # line1, = ax2.plot(df['bandwidth'], battery_plot_data, color=sns.xkcd_rgb["pale red"], linestyle='-',
    #                   label='Battery life')
    # p1 = ax2.scatter(df['bandwidth'], battery_plot_data, color=sns.xkcd_rgb["pale red"], marker='o', s=30,
    #                  label='Battery life')

    # note = ax2.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")

    # for i, j, d in zip(df['bandwidth'], battery_plot_data, df["device"]):
    #     ax2.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])
    ######################################## End of Battery life

    # if min(battery_plot_data) > 0:
    #     ax2.set_ylim(ymin=0, ymax=(max(battery_plot_data + 2)))
    # ax2.set_ylabel("Battery life", fontsize=12)
    # plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f x'))
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Energy (J)", fontsize=12)
    # ax1.set_title(f"{config}", fontsize=14)

    # Set colors for y-axis tags
    # ax2.yaxis.label.set_color(line1.get_color())
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    # ax2.tick_params(axis='y', colors=line1.get_color())
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    plt.legend(handles=[l1, l3, b0, b1], loc="lower right", prop={'size': 6})
    plt.grid()
    plt.savefig(f"{POWER_MODE}/{config}-2G.png", bbox_inches='tight', dpi=100)


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
         # 'yolos-agx'
    ]

    for config in tqdm(configs):
        draw(config)

    # print(sum(avg_battery)/len(avg_battery))