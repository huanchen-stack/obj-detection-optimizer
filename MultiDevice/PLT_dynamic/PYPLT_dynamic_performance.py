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
    fig, (ax2, ax1)= plt.subplots(2,1)
    fig.set_size_inches(8, 3.5)
    fig.set_dpi(100)
######################
    df_file = pd.read_csv(f"{config}.csv")
    x1_list = []
    for i in range(len(df_file['bandwidth'])):
        x1_list.append(i)

    plot_data = df_file['bandwidth']
    line1, = ax2.plot(x1_list, plot_data,
                     color=sns.xkcd_rgb["denim blue"],
                     linestyle='-',
                     label='bandwidth',
                      linewidth=1.5)
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=12)
    # Set colors for y-axis tags
    ax2.yaxis.label.set_color('black')
    # Set colors for y-axis marks
    ax2.tick_params(axis='y', colors='black')
    # Set legends
    # plt.legend(handles=[line1], bbox_to_anchor=(0.5, 1.25), loc='upper center')
    plt.grid()
    ax2.get_xaxis().set_visible(False)
######################


    df_file = pd.read_csv(f"{config}.csv")
    x1_list = []
    for i in range(len(df_file['bandwidth'])):
        x1_list.append(i)

    plot_data = df_file['optimizer']
    device_nums = df_file['device']

    line1, = ax1.plot(x1_list, plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='optimization result',
                      linewidth=1.5)
    ax1.set_ylabel("Perf. Improv. (%)", fontsize=12)
    ax1.set_ylim(ymax=(max(plot_data + 4)))
    # method 1: plot every #devices
    # loc = [0]
    # for d in range(1, len(device_nums)-1):
    #     if device_nums[d] != device_nums[d-1]:
    #         plt.axvline(x=d-0.5, linestyle='--')
    #         plt.text(x=(d + loc[-1]) * 0.5, y=2,
    #                  s=f'{device_nums[d-1]} devices' if device_nums[d-1] > 1 else 'local',
    #                  horizontalalignment='center')
    #         loc.append(d)
    # plt.text(x=(len(device_nums)-1 + loc[-1]) * 0.5, y=2,
    #          s=f'{device_nums[len(device_nums)-1]} devices' if device_nums[len(device_nums)-1] > 1 else 'local',
    #          horizontalalignment='center')
    # method 2: plot the max #devices
    loc_max = []
    loc_min_1 = []
    loc_min_2 = []
    max_found = False
    max_num = max(device_nums)
    min_num = min(device_nums)
    y = max(plot_data + 2)
    for d in range(1, len(device_nums) - 1):
        if device_nums[d] == max_num:
            loc_max.append(d)
            max_found = True
        elif device_nums[d] == min_num:
            if not max_found:
                loc_min_1.append(d)
            else:
                loc_min_2.append(d)
    plt.axvline(x=loc_max[0] - 0.5, linestyle='--')
    plt.axvline(x=loc_max[-1] + 0.5, linestyle='--')
    plt.axvline(x=loc_min_1[-1] + 0.5, linestyle='--')
    plt.axvline(x=loc_min_2[0] - 0.5, linestyle='--')
    plt.text(x=(loc_max[0] + loc_max[-1])/2, y=max(plot_data + 2),
             s=f'{max_num} partitions',
             horizontalalignment='center')
    plt.text(x=(loc_min_1[-1])/2 - 1.2, y=max(plot_data + 2),
             s=f'{min_num} partitions' if min_num > 1 else '1 partition',
             horizontalalignment='center')
    plt.text(x=(len(device_nums) + loc_min_2[0])/2 + 1.2, y=max(plot_data + 2),
             s=f'{min_num} partitions' if min_num > 1 else '1 partition',
             horizontalalignment='center')
    plt.text(x=(loc_max[-1] + loc_min_2[0]) / 2, y=max(plot_data + 2),
             s=f'{min_num + 1} - {max_num - 1} partitions',
             horizontalalignment='center')
    plt.text(x=(loc_max[0] + loc_min_1[-1]) / 2, y=max(plot_data + 2),
             s=f'{min_num + 1} - {max_num - 1} partitions',
             horizontalalignment='center')

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    # plt.legend(handles=[line1], bbox_to_anchor=(0.5, 1.25), loc='upper center')
    plt.grid()
    ax1.get_xaxis().set_visible(False)


    plt.savefig(f"plots/{config}-performance.pdf", bbox_inches='tight', dpi=100)
    plt.savefig(f"plots/{config}-performance.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        # 'faster-agx',
        # 'yolox-agx',
        'yolox-nano',
    ]

    for config in tqdm(configs):
        draw(config)
