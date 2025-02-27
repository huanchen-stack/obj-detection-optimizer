import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER, POWER_MODE
from PLT_energy.PYPLT_energy_2G import baseE

COMPARE_NS_Original = True

def draw(config):

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 2.2)
    df_file = pd.read_csv(f"../data/final_data/sufficient_memory/{config}.csv")
    x1_list = []
    for i in df_file['bandwidth']:
        x1_list.append(i)
        
    plot_data = df_file['optimizer']

    line1 = ax1.plot(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='Optimization speed up' if not COMPARE_NS_Original else 'SN (this work)')
    p1 = ax1.scatter(df_file['bandwidth'], plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     marker='o',
                     s=30,
                     label='Optimization speed up' if not COMPARE_NS_Original else 'SN (this work)')

    if COMPARE_NS_Original:
        df_file_origin = pd.read_csv(f"../data/final_data/original_NS_sufficient_memory/{config}.csv")
        x2_list = []
        for i in df_file_origin['bandwidth']:
            x2_list.append(i)

        plot_data_origin = df_file_origin['optimizer']

        line2 = ax1.plot(df_file_origin['bandwidth'], plot_data_origin,
                         color=sns.xkcd_rgb["denim blue"],
                         linestyle='-',
                         label='Neurosurgoen (%)')
        p2 = ax1.scatter(df_file_origin['bandwidth'], plot_data_origin,
                         color=sns.xkcd_rgb["denim blue"],
                         marker='s',
                         s=30,
                         label='Neurosurgoen')
        # for i, j, d in zip(df_file['bandwidth'], plot_data_origin, df_file_origin["device"]):
        #     ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])
        # note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#partitions")


    # for i, j, d in zip(df_file['bandwidth'], plot_data, df_file["device"]):
    #     ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#partitions")
    # baseline = ax1.hlines(y=baseBattery[config]/baseE[config], color=sns.xkcd_rgb["denim blue"], linestyle='-', xmin=x1_list[0], xmax=x1_list[-1], label="base battery life")

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Perf. Improv. (%)", fontsize=12)
    # ax1.set_title(f"{config}", fontsize=14)
    if min(plot_data) > 0:
        # ax1.set_ylim(ymin=0, ymax=(max(plot_data + 2)))
        pass

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    if COMPARE_NS_Original:
        plt.legend(handles=[p1, p2], loc='best')
        # plt.legend(handles=[p1, p2, note], bbox_to_anchor=(0.5, 1.5), loc='upper center', ncol=3)
        pass
    else:
        # plt.legend(handles=[p1, note], loc='lower right')
        pass
    plt.grid()
    # plt.title('4GB')
    if COMPARE_NS_Original:
        plt.savefig(f"compare/{config}.png", bbox_inches='tight', dpi=100)
        plt.savefig(f"compare/{config}.pdf", bbox_inches='tight', dpi=100)
    else:
        plt.savefig(f"{POWER_MODE}/{config}.png", bbox_inches='tight', dpi=100)
        plt.savefig(f"{POWER_MODE}/{config}.pdf", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        'faster-agx',
        'faster-nano',
        'yolor-agx',
        'yolor-nano',
        'yolox-agx',
        'yolox-nano',
        'yolov4-agx',
        'yolov4-nano'
        #  'yolos-agx'
    ]

    for config in tqdm(configs):
        draw(config)
