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

    plot_data = df_file['optimizer']
    device_nums = df_file['device']

    line1, = ax1.plot(x1_list, plot_data,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='optimization result',
                      linewidth=1.5)
    ax1.set_ylabel("Performance (%)", fontsize=12)
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
             s=f'{max_num} devices',
             horizontalalignment='center')
    plt.text(x=(loc_min_1[-1])/2, y=max(plot_data + 2),
             s=f'{min_num} devices' if min_num > 1 else 'local',
             horizontalalignment='center')
    plt.text(x=(len(device_nums) + loc_min_2[0])/2, y=max(plot_data + 2),
             s=f'{min_num} devices' if min_num > 1 else 'local',
             horizontalalignment='center')
    plt.text(x=(loc_max[-1] + loc_min_2[0]) / 2, y=max(plot_data + 2),
             s=f'{min_num + 1} - {max_num - 1} devices',
             horizontalalignment='center')
    plt.text(x=(loc_max[0] + loc_min_1[-1]) / 2, y=max(plot_data + 2),
             s=f'{min_num + 1} - {max_num - 1} devices',
             horizontalalignment='center')

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    plt.legend(handles=[line1], bbox_to_anchor=(0.5, 1.25), loc='upper center')
    plt.grid()
    ax1.get_xaxis().set_visible(False)


    plt.savefig(f"plots/{config}-performance.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = [
        # 'faster-agx',
        'yolox-agx',
    ]

    for config in tqdm(configs):
        draw(config)
