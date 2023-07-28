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

COMPARE_NS_Original = True


def draw(config):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 2)
    df_file = pd.read_csv(f"{config}.csv")
    x1_list = []
    for i in range(len(df_file['bandwidth'])):
        x1_list.append(i)

    plot_data = df_file['bandwidth']

    line1, = ax1.plot(x1_list, plot_data,
                     color=sns.xkcd_rgb["denim blue"],
                     linestyle='-',
                     label='bandwidth',
                      linewidth=1.5)
    # p1 = ax1.scatter(x1_list, plot_data,
    #                  color=sns.xkcd_rgb["pale red"],
    #                  marker='o',
    #                  s=30,
    #                  label='bandwidth')
    # ax1.set_ylim(ymin=0)
    ax1.set_ylabel("Mbps", fontsize=12)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    plt.legend(handles=[line1], bbox_to_anchor=(0.5, 1.25), loc='upper center')
    plt.grid()
    ax1.get_xaxis().set_visible(False)

    plt.savefig(f"plots/bandwidth.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        # 'faster-agx',
        'yolox-agx',
    ]

    for config in tqdm(configs):
        draw(config)
