import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER, POWER_MODE
from PLT_energy.PYPLT_energy import baseE


def draw(config):

    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 3)
    df_file = pd.read_csv(f"../data/{config}.csv")
    x1_list = []
    for i in df_file['bandwidth']:
        x1_list.append(i)
        
    plot_data = df_file['optimizer']

    line1 = ax1.plot(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='Optimization speed up')
    p1 = ax1.scatter(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     marker='o',
                     s=30,
                     label='Optimization speed up')

    for i, j, d in zip(df_file['bandwidth'], plot_data, df_file["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")
    # baseline = ax1.hlines(y=baseBattery[config]/baseE[config], color=sns.xkcd_rgb["denim blue"], linestyle='-', xmin=x1_list[0], xmax=x1_list[-1], label="base battery life")

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Optimization speed up (%)", fontsize=12)
    # ax1.set_title(f"{config}", fontsize=14)
    if min(plot_data) > 0:
        ax1.set_ylim(ymin=0, ymax=(max(plot_data + 2)))

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    plt.legend(handles=[p1, note], loc='lower right')
    plt.grid()
    plt.savefig(f"{POWER_MODE}/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = OPT_WRAPPER.configs

    for config in tqdm(configs):
        draw(config)