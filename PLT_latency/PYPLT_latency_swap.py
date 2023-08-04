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

# BANDWIDTH = "4GB"
# READ_SPEED = 1280
READ_SPEED = 190

baseline = {
    "2GB": {
        "yolos-agx": 12321 / READ_SPEED,
        "yolox-agx": 5091.6 / READ_SPEED,
        "yolor-agx": 3250 / READ_SPEED,
        "yolov4-agx": 3547 / READ_SPEED,
        "faster-agx": 3055.5 / READ_SPEED,
        "yolox-nano": 5091.6 / READ_SPEED,
        "yolor-nano": 3250 / READ_SPEED,
        "yolov4-nano": 3547 / READ_SPEED,
        "faster-nano": 3055.5 / READ_SPEED,
    },
    "4GB": {
        "yolos-agx": 10504 / READ_SPEED,
        "yolox-agx": 2800 / READ_SPEED,
        "yolor-agx": 1206 / READ_SPEED,
        "yolov4-agx": 1607 / READ_SPEED,
        "faster-agx": 715 / READ_SPEED,
        "yolox-nano": 2800 / READ_SPEED,
        "yolor-nano": 1206 / READ_SPEED,
        "yolov4-nano": 1607 / READ_SPEED,
        "faster-nano": 715 / READ_SPEED,
    }
}

benchmarks = {
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
}

def draw(config):

    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 3)
    df_file_2GB = pd.read_csv(f"../data/2GB/{config}.csv")
    df_file_4GB = pd.read_csv(f"../data/4GB/{config}.csv")
    x1_list = []
    x2_list = []
    for i in df_file_2GB['bandwidth']:
        x1_list.append(i)
        x2_list.append(i)
        
    plot_data_2GB = df_file_2GB['optimizer']
    plot_data_4GB = df_file_4GB['optimizer']
    for i in range(len(plot_data_2GB)):
        plot_data_2GB[i] = (baseline['2GB'][config] + benchmarks[config]) / (benchmarks[config] - plot_data_2GB[i] * 0.01 * benchmarks[config])
        plot_data_4GB[i] = (baseline['4GB'][config] + benchmarks[config]) / (benchmarks[config] - plot_data_2GB[i] * 0.01 * benchmarks[config])

    line1 = ax1.plot(df_file_2GB['bandwidth'], plot_data_2GB,
                     color=sns.xkcd_rgb["denim blue"],
                     linestyle='-',
                     label='Optimization speed up (times)')
    p1 = ax1.scatter(df_file_2GB['bandwidth'], plot_data_2GB,
                     color=sns.xkcd_rgb["denim blue"],
                     marker='o',
                     s=30,
                     label='Optimization speed up (times)')

    line2 = ax1.plot(df_file_4GB['bandwidth'], plot_data_4GB,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='Optimization speed up (times)')
    p2 = ax1.scatter(df_file_4GB['bandwidth'], plot_data_4GB,
                     color=sns.xkcd_rgb["pale red"],
                     marker='o',
                     s=30,
                     label='Optimization speed up (times)')

    for i, j, d in zip(df_file_2GB['bandwidth'], plot_data_2GB, df_file_2GB["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    for i, j, d in zip(df_file_4GB['bandwidth'], plot_data_4GB, df_file_4GB["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")
    # baseline = ax1.hlines(y=baseBattery[config]/baseE[config], color=sns.xkcd_rgb["denim blue"], linestyle='-', xmin=x1_list[0], xmax=x1_list[-1], label="base battery life")

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Optimization speed up (times)", fontsize=12)
    ax1.yaxis.set_major_formatter('{x:1.1f}x')
    # ax1.set_title(f"{config}", fontsize=14)
    # if min(plot_data) > 0:
    ax1.set_ylim(ymin=0, ymax=(max(plot_data_2GB + 2)))

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    # if COMPARE_NS_Original:
    #     # plt.legend(handles=[p1, p2, note], loc='lower right')
    #     # plt.legend(handles=[p1, p2, note], bbox_to_anchor=(0.5, 1.5), loc='upper center', ncol=3)
    #     pass
    # else:
    #     plt.legend(handles=[p1, note], loc='lower right')
    # plt.legend(handles=[p1, note], loc='lower right')

    plt.grid()
    # plt.title('4GB')

    plt.savefig(f"swap/{READ_SPEED}/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        # 'faster-agx',
        'faster-nano',
        # 'yolor-agx',
        'yolor-nano',
        # 'yolox-agx',
        'yolox-nano',
        # 'yolov4-agx',
        'yolov4-nano'
        #  'yolos-agx'
    ]

    for config in tqdm(configs):
        draw(config)
