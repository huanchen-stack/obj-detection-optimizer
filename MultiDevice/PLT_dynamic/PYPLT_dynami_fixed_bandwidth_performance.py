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
    fig.set_size_inches(4, 2)
    df_file = pd.read_csv(f"../data/{config}.csv")
    x1_list = []
    for i in range(len(df_file['bandwidth'])):
        x1_list.append(i)

    plot_data = df_file['optimizer']

    # data processing
    baseline = plot_data[0]
    for i in range(len(plot_data)):
        plot_data[i] = (baseline - plot_data[i]) / baseline + 1

    device_nums = df_file['device']

    # line1, = ax1.plot(x1_list, plot_data,
    #                  color=sns.xkcd_rgb["pale red"],
    #                  linestyle='-',
    #                  label='optimization result')
    p1 = ax1.bar(x1_list, plot_data,
                     color=sns.xkcd_rgb["pale red"], width=0.5,
                     # marker='o',
                     # s=30,
                     label='Speed')
    ax1.set_ylabel("End-to-end speed", fontsize=12)
    ax1.set_xlabel("Scenarios", fontsize=12)
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
    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')
    ax1.xaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')
    ax1.tick_params(axis='x', colors='black')
    plt.locator_params(axis='x', nbins=4)
    ax1.yaxis.set_major_formatter('{x:1.1f}x')

    # Set legends
    plt.legend(handles=[p1], bbox_to_anchor=(0.15, 1.3), loc='upper center')
    # plt.legend(handles=[p1], loc='best')
    plt.grid()
    plt.ylim([0, 2.8])
    plt.xlim([-0.5, 3.5])
    # ax1.get_xaxis().set_visible(True)

    plt.savefig(f"../plots/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        'yolor',
    ]

    for config in tqdm(configs):
        draw(config)
