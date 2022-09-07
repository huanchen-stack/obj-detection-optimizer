import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER

def draw(config):
    baseBattery = {
        'faster-agx':   1,
        'yolor-agx':    1,
        'yolox-agx':    1,
        'yolov4-agx':   1,
        'faster-nano':  1,
        'yolor-nano':   1,
        'yolox-nano':   1,
        'yolov4-nano':  1,
    }

    baseE = {
        'faster-agx': 0,
        'yolor-agx': 0,
        'yolox-agx': 0,
        'yolov4-agx': 0,
        'faster-nano': 14213.7446,
        'yolor-nano': 10447.9559,
        'yolox-nano': 13460.1792,
        'yolov4-nano': 9055.3908,
    }

    fig, ax1 = plt.subplots()
    df_file = pd.read_csv(f"../data/{config}.csv")
    x1_list = []
    for i in df_file['bandwidth']:
        x1_list.append(i)

    line1 = ax1.plot(df_file['bandwidth'], baseBattery[config]*df_file['device']/(baseE[config]+df_file['energy']),
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='battery life')
    p1 = ax1.scatter(df_file['bandwidth'], baseBattery[config]*df_file['device']/(baseE[config]+df_file['energy']),
                     color=sns.xkcd_rgb["pale red"],
                     marker='o',
                     s=30,
                     label='battery life')

    for i, j, d in zip(df_file['bandwidth'], baseBattery[config]*df_file['device']/(baseE[config]+df_file['energy']), df_file["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")
    baseline = ax1.hlines(y=baseBattery[config]/baseE[config], color=sns.xkcd_rgb["denim blue"], linestyle='-', xmin=x1_list[0], xmax=x1_list[-1], label="base battery life")

    # ax2.set_ylim([0, 15])
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Battery life", fontsize=12)
    ax1.set_title(f"{config}", fontsize=14)
    ax1.ticklabel_format(style='sci', scilimits=(-3, 5))

    # 双Y轴标签颜色设置
    ax1.yaxis.label.set_color('black')

    # 双Y轴刻度颜色设置
    ax1.tick_params(axis='y', colors='black')

    # 图例设置
    plt.legend(handles=[p1, note, baseline], loc=(1.04, 0))
    plt.grid()
    plt.savefig(f"{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = OPT_WRAPPER.configs

    for config in tqdm(configs):
        draw(config)