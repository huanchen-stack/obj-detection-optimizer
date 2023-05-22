import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from opt_wrapper import OPT_WRAPPER, POWER_MODE
import matplotlib.ticker as mticker


baseE = {
    "0": {
        'faster-agx': 8639.9460,
        'yolor-agx': 7679.2450,
        'yolox-agx': 7179.3893,
        'yolov4-agx': 4213.3583,
        'faster-nano': 14213.7446,
        'yolor-nano': 10447.9559,
        'yolox-nano': 13460.1792,
        'yolov4-nano': 9055.3908,
    },
    "1": {
        'faster-agx': 8724.0558,
        'yolor-agx': 9049.7275,
        'yolox-agx': 24857.6348,
        'yolov4-agx': 7468.0064,
        'faster-nano': 12406.4492,
        'yolor-nano': 9158.5191,
        'yolox-nano': 38119.5639,
        'yolov4-nano': 10734.7091,
    }

}

baseBattery = {
    'faster-agx': 1,
    'yolor-agx': 1,
    'yolox-agx': 1,
    'yolov4-agx': 1,
    'faster-nano': 1,
    'yolor-nano': 1,
    'yolox-nano': 1,
    'yolov4-nano': 1,
}

avg_battery = []

def draw(config):
    power = "0"
    fig, ax1 = plt.subplots()
    fig.set_size_inches(6, 3)
    df = pd.read_csv(f"../data/{config}.csv")
    if len(df['bandwidth']) == 0:
        print("No results printed")
        return

    x1_list = []
    for i in df['bandwidth']:
        x1_list.append(i)

    base = 0
    if baseE[power][config] != 0:
        base = baseE[power][config]
    battery_plot_data = baseBattery[config] * df['device'] / (baseE[POWER_MODE][config] + df['energy']) / (
            baseBattery[config] / baseE[POWER_MODE][config])

    # avg_battery.append(battery_plot_data[len(battery_plot_data)-1])
    avg_battery.append(battery_plot_data[0])

    w = (df['bandwidth'][len(df['bandwidth']) - 1] - df['bandwidth'][0]) / len(df) * 0.5
    b0 = ax1.bar(x1_list, base, width=w, label='Energy: Computation',
                 color=sns.xkcd_rgb["denim blue"])
    b1 = ax1.bar(x1_list, df['energy'], width=w, label='Energy: Communication',
                 bottom=base, color=sns.xkcd_rgb["maize"])

    ax2 = ax1.twinx()

    line1, = ax2.plot(df['bandwidth'], battery_plot_data, color=sns.xkcd_rgb["pale red"], linestyle='-',
                      label='Battery life')
    p1 = ax2.scatter(df['bandwidth'], battery_plot_data, color=sns.xkcd_rgb["pale red"], marker='o', s=30,
                     label='Battery life')

    note = ax2.scatter([], [], marker='$1$', color=sns.xkcd_rgb["green"], label="#device needed for optimization")

    for i, j, d in zip(df['bandwidth'], battery_plot_data, df["device"]):
        ax2.annotate('%s' % d, xy=(i, j), xytext=(-7, 3), textcoords='offset points', color=sns.xkcd_rgb["green"])

    if min(battery_plot_data) > 0:
        ax2.set_ylim(ymin=0, ymax=(max(battery_plot_data + 2)))
    ax2.set_ylabel("Battery life", fontsize=12)
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f x'))
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Energy (mJ)", fontsize=12)
    # ax1.set_title(f"{config}", fontsize=14)

    # Set colors for y-axis tags
    ax2.yaxis.label.set_color(line1.get_color())
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax2.tick_params(axis='y', colors=line1.get_color())
    ax1.tick_params(axis='y', colors='black')

    # Set legends
    plt.legend(handles=[p1, b0, b1, note], loc="lower right", prop={'size': 6})
    plt.grid()
    plt.savefig(f"{POWER_MODE}/{config}.png", bbox_inches='tight', dpi=100)


if __name__ == "__main__":
    configs = OPT_WRAPPER.configs

    for config in tqdm(configs):
        draw(config)

    print(sum(avg_battery)/len(avg_battery))