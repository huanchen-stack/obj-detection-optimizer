import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER

def draw(config):
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
    df_year = pd.read_csv(f"{config}.csv")
    x1_list = []
    for i in df_year['bandwidth']:
        x1_list.append(i)

    base = 0
    if baseE[config] != 0:
        base = baseE[config]

    w = (df_year['bandwidth'][len(df_year['bandwidth'])-1] - df_year['bandwidth'][0]) / len(df_year) * 0.5
    b0 = ax1.bar(x1_list, base, width=w, label='Energy: Computation',
                 color=sns.xkcd_rgb["denim blue"])
    b1 = ax1.bar(x1_list, df_year['energy'], width=w, label='Energy: Communication',
                 bottom=base, color=sns.xkcd_rgb["maize"])

    ax2 = ax1.twinx()

    line1, = ax2.plot(df_year['bandwidth'], df_year['optimizer'], color=sns.xkcd_rgb["pale red"], linestyle='-',
                    label='optimizer performance')
    p1 = ax2.scatter(df_year['bandwidth'], df_year['optimizer'], color=sns.xkcd_rgb["pale red"], marker='o', s=30,
                    label='optimizer performance')

    for i, j, d in zip(df_year['bandwidth'], df_year["optimizer"], df_year["device"]):
        ax2.annotate('%s' % d, xy=(i, j), xytext=(-13, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax2.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")

    # ax2.set_ylim([0, 15])
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Energy (mJ)", fontsize=12)
    ax2.set_ylabel("Optimization (%)", fontsize=12)
    ax1.set_title(f"{config}", fontsize=14)

    # 双Y轴标签颜色设置
    ax2.yaxis.label.set_color(line1.get_color())
    ax1.yaxis.label.set_color('black')

    # 双Y轴刻度颜色设置
    ax2.tick_params(axis='y', colors=line1.get_color())
    ax1.tick_params(axis='y', colors='black')

    # 图例设置
    plt.legend(handles=[p1, b0, b1, note])
    plt.grid()
    plt.savefig(f"{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = OPT_WRAPPER.configs

    for config in tqdm(configs):
        draw(config)