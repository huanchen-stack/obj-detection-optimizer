import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

COMPARE_NS_Original = True


def draw(config):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 2.2)
    df_file1 = pd.read_csv(f"../data/final_data/sufficient_memory/{config}-nano.csv")
    df_file_origin1 = pd.read_csv(f"../data/final_data/original_NS_sufficient_memory/{config}-nano.csv")

    df_file2 = pd.read_csv(f"../data/final_data/sufficient_memory/{config}-agx.csv")
    df_file_origin2 = pd.read_csv(f"../data/final_data/original_NS_sufficient_memory/{config}-agx.csv")

    plot_data1 = df_file1['optimizer'] #- df_file_origin1['optimizer']
    plot_data2 = df_file2['optimizer'] #- df_file_origin2['optimizer']

    line1 = ax1.plot(df_file1['bandwidth'], plot_data1,
                     color=sns.xkcd_rgb["russet"],
                     linestyle='-')
    p1 = ax1.scatter(df_file1['bandwidth'], plot_data1,
                     color=sns.xkcd_rgb["russet"],
                     marker='o',
                     s=30)

    line2 = ax1.plot(df_file2['bandwidth'], plot_data2,
                     color=sns.xkcd_rgb["pinky purple"],
                     linestyle='-')
    p2 = ax1.scatter(df_file2['bandwidth'], plot_data2,
                     color=sns.xkcd_rgb["pinky purple"],
                     marker='s',
                     s=30)

    for i, j, d in zip(df_file1['bandwidth'], plot_data1, df_file1["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    for i, j, d in zip(df_file2['bandwidth'], plot_data2, df_file2["device"]):
        ax1.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    note = ax1.scatter([], [], marker='$N$', color=sns.xkcd_rgb["green"], label="#partions for optimization")

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Latency Improv. (%)", fontsize=12)

    ax1.set_ylim(ymin=0, ymax=max(max(plot_data1), max(plot_data2)) * 1.1)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    labels = ["Jetson-Nano", "Jetson-AGX", "#partitions"]

    # Set legends
    if config == 'yolor':
        plt.legend(handles=[p1, p2, note], labels=labels, loc='best')
    else:
        plt.legend(handles=[p2, p1, note], labels=labels, loc='best')
    plt.grid()
    # plt.title('4GB')

    plt.savefig(f"compare-old/{config}.pdf", bbox_inches='tight', dpi=100)
    plt.savefig(f"compare-old/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        'faster',
        'yolor',
        'yolox',
        'yolov4',
    ]

    for config in tqdm(configs):
        draw(config)
