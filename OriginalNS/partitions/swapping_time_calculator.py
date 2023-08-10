import pandas as pd
import os
import seaborn as sns

from matplotlib import pyplot as plt, ticker

# READ_SPEED = 1280
READ_SPEED = 190

benchmarks = {
    "0": {
        'faster-agx': 0.349021,
        'faster-nano': 1.967904,
        'faster-clarity32': 0.063555,
        'yolor-agx': 0.1736289,
        'yolor-nano': 1.458861,
        'yolox-agx': 2.6009,
        'yolox-nano': 1.6617,
        'yolov4-agx': 0.1969,
        'yolov4-nano': 1.1422,
    },
    "1": {
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
    },
}

# Create a custom formatter to format y-axis labels
def custom_format(x, pos):
    if x.is_integer():
        return f'{int(x)}x'
    else:
        return f'{x:.1f}'

def calculate(config, mem):
    res = {}
    for b in range(250, 2500, 250):
        device_dict, mem_dict = load_partition_and_mem(config, b)
        res[b] = 0
        for i in range(0,2):
            sum = 0
            for layer in device_dict[i]:
                sum += mem_dict[layer]
            if sum > mem:
                res[b] += (sum - mem)/READ_SPEED
            else:
                res[b] += 0
    res_file = open(os.path.join(f"{config}/swap_time.csv"), "w")
    res_file.write(f"bandwidth,time\n")
    for b,t in res.items():
        res_file.write(f"{b},{t}\n")
    res_file.close()

def add(config, mem):
    swap_time = pd.read_csv(os.path.join(f"{config}/swap_time.csv"))
    original_data = pd.read_csv(os.path.join(f"../data/{config}.csv"))
    for i in range(len(swap_time)):
        original_time = (100 - original_data['optimizer'][i]) * benchmarks['1'][config] / 100
        original_data['optimizer'][i] = 100 - 100 * (swap_time['time'][i] + original_time / benchmarks['1'][config])
    original_data.to_csv(os.path.join(f"../compare_with_opt/{int(mem/1024)}GB/{int(READ_SPEED)}/{config}.csv"), sep=',', index=False)

def compare(config, mem):
    opt_res = pd.read_csv(os.path.join(f"../../data/final_data/mem_constrained/{int(mem/1024)}GB/{config}.csv"))
    NS_res = pd.read_csv(os.path.join(f"../compare_with_opt/{int(mem/1024)}GB/{int(READ_SPEED)}/{config}.csv"))
    res = []
    for i in range(len(opt_res)):
        opt = (100 - opt_res['optimizer'][i]) * benchmarks['1'][config] / 100
        NS = (100 - NS_res['optimizer'][i]) * benchmarks['1'][config] / 100
        res.append(NS/opt)
    return res


def load_partition_and_mem(config, bandwidth):
    device_dict = {
        0: [],
        1: [],
    }
    mem_dict = {}
    part_list = pd.read_csv(f"{config}/part-{bandwidth}.csv").values.tolist()
    for entry in part_list:
        layer = entry[0]
        part = entry[1]
        device_dict[part].append(layer)
    prof_list = pd.read_csv(f"../../testcases/{config}/1/prof.csv").values.tolist()
    for entry in prof_list:
        layer = entry[0]
        mem = entry[3]
        mem_dict[layer] = mem
    return device_dict, mem_dict


def draw(config, res):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(4, 2.2)
    bandwidth = [*range(250, 2500, 250)]
    
    plot_data_2GB = res[config]['2GB']
    plot_data_4GB = res[config]['4GB']

    line1 = ax1.plot(bandwidth, plot_data_2GB,
                     color=sns.xkcd_rgb["denim blue"],
                     linestyle='-',
                     label='Optimization speed up (times)')
    p1 = ax1.scatter(bandwidth, plot_data_2GB,
                     color=sns.xkcd_rgb["denim blue"],
                     marker='o',
                     s=30,
                     label='2GB')

    line2 = ax1.plot(bandwidth, plot_data_4GB,
                     color=sns.xkcd_rgb["pale red"],
                     linestyle='-',
                     label='Opt. speed up (times)')
    p2 = ax1.scatter(bandwidth, plot_data_4GB,
                     color=sns.xkcd_rgb["pale red"],
                     marker='s',
                     s=30,
                     label='4GB')

    # ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Bandwidth (Mbps)", fontsize=12)
    ax1.set_ylabel("Speedup", fontsize=12)

    if READ_SPEED == 190:
        ax1.yaxis.set_major_locator(ticker.FixedLocator([1, 5, 10, 15, 20, 25]))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(custom_format))
    ax1.set_ylim(ymin=1, ymax=max(max(plot_data_2GB), max(plot_data_4GB)) * 1.1)

    # Set colors for y-axis tags
    ax1.yaxis.label.set_color('black')

    # Set colors for y-axis marks
    ax1.tick_params(axis='y', colors='black')

    plt.legend(handles=[p1, p2], loc='best', prop={'size': 8}, ncol=3)
    plt.grid()

    plt.savefig(f"../compare_with_opt/swap/{READ_SPEED}/{config}.png", bbox_inches='tight', dpi=100)
    plt.savefig(f"../compare_with_opt/swap/{READ_SPEED}/pdfs/{config}.pdf", bbox_inches='tight', dpi=100)


if __name__ == '__main__':
    configs = [
        # 'faster-agx',
        'faster-nano',
        # 'yolor-agx',
        'yolor-nano',
        # 'yolox-agx',
        'yolox-nano',
        # 'yolov4-agx',
        'yolov4-nano',
        # 'yolos-agx'
    ]
    res = {
        'yolov4-nano': {},
        'yolox-nano': {},
        'yolor-nano': {},
        'faster-nano': {}
    }
    for config in configs:
        calculate(config, 2*1024)
        add(config, 2*1024)
        res[config]['2GB'] = compare(config, 2*1024)
    for config in configs:
        calculate(config, 4*1024)
        add(config, 4*1024)
        res[config]['4GB'] = compare(config, 4*1024)

    for config in configs:
        print(max(res[config]['2GB']))
        print(max(res[config]['4GB']))
        draw(config, res)



