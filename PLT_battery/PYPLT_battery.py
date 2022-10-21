import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER, POWER_MODE
from PLT_energy.PYPLT_energy import baseE

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

    fig, ax1 = plt.subplots()
    df_file = pd.read_csv(f"../data/{config}.csv")
    x1_list = []
    for i in df_file['bandwidth']:
        x1_list.append(i)
        
    plot_data = baseBattery[config]*df_file['device']/(baseE[POWER_MODE][config]+df_file['energy']) / (baseBattery[config]/baseE[POWER_MODE][config])

    line1 = ax1.plot(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='battery life')
    p1 = ax1.scatter(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     marker='o',
                     s=30,
                     label='battery life')

    for i, j, d in zip(df_file['bandwidth'], plot_data, df_file["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")
    # baseline = ax1.hlines(y=baseBattery[config]/baseE[config], color=sns.xkcd_rgb["denim blue"], linestyle='-', xmin=x1_list[0], xmax=x1_list[-1], label="base battery life")

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Battery life", fontsize=12)
    ax1.set_title(f"{config}", fontsize=14)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f x'))

    # 双Y轴标签颜色设置
    ax1.yaxis.label.set_color('black')

    # 双Y轴刻度颜色设置
    ax1.tick_params(axis='y', colors='black')

    # 图例设置
    plt.legend(handles=[p1, note], loc=(1.04, 0))
    plt.grid()
    plt.savefig(f"{POWER_MODE}/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = OPT_WRAPPER.configs

    for config in tqdm(configs):
        draw(config)