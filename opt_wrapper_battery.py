from queue import Empty
from random import expovariate
import re
from optimizer_battery import Optimizer
from simulatorv2 import Simulator
import os
from tqdm import tqdm


class OPT_WRAPPER(object):
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

    configs = [
        # 'faster-agx',
        # 'faster-nano',
        # 'yolor-agx',
        # 'yolor-nano',
        # 'yolox-agx',
        # 'yolox-nano',
        'yolov4-agx',
        'yolov4-nano'
    ]
    benchmarks = {
        "0": {
            'faster-agx': 0.349021,
            'faster-nano': 1.967904,
            'faster-clarity32': 0.063555,
            'yolor-agx': 0.1736289,
            'yolor-nano': 1.458861,
            # 'yolox-agx': 2.6009,
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
        },

    }
    bandwidths = {
        # 'agx': [
        #     750, 1000, 1250, 1500, 1750,
        #     2000, 2250, 2500, 2750, 3000, float('inf'),
        # ],
        # 'nano': [
        #     375, 500, 625, 750, 875,
        #     1000, 1250, 1500, 1750, 2000, 3000,
        # ],
        # 'agx': [*range(900, 3400, 100)],
        'agx':
            {'yolox': [*range(250, 4500, 150)],
             'yolor': [*range(250, 4500, 150)],
             'yolov4': [*range(250, 8000, 250)],
             'faster': [*range(900, 3400, 100)]},
        # 'nano': [*range(375, 1500, 125)],  # good graph
        'nano':
            {'yolox': [*range(250, 4500, 150)],
             'yolor': [*range(250, 4500, 150)],
             'yolov4': [*range(250, 8000, 250)],
             'faster': [*range(900, 3400, 100)]},
    }

    def __init__(self, config, bandwidth_list=None, threshold=0.99):
        super().__init__()
        self.config = config
        self.benchmark = OPT_WRAPPER.benchmarks["0"][self.config]

        if bandwidth_list is None:
            self.bandwidth_list = OPT_WRAPPER.bandwidths[config.split('-')[1]][config.split('-')[0]]
        else:
            self.bandwidth_list = bandwidth_list
        self.bandwidth_list = [bw * 0.125 for bw in self.bandwidth_list]  # turn to MBps

        self.iterations_default = 5
        self.num_devices_max = 7
        self.threshold = threshold

        self.opt_num_devices = []
        self.opt_speedup_rate = []
        self.payload = []
        self.args = []

    def get_path(self):
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"testcases/{self.config}")
        self.dep = os.path.join(path, "dep.csv")
        self.prof = os.path.join(path, "prof.csv")
        self.priority = os.path.join(path, "priority.csv")
        self.part = os.path.join(path, "part.csv")


    def optimize_once(self, bandwidth, num_devices, reverse0, reverse1, iterations=None, power_setting=0):
        iterations = iterations if iterations is not None else self.iterations_default
        opt = Optimizer(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * num_devices,
            bandwidth=bandwidth,
            ignore_latency=False,  # always False
            iterations=iterations,
            benchmark_energy=self.baseE[str(power_setting)][self.config],
            reverse0=reverse0,
            reverse1=reverse1,
            dir=f"testcases/{self.config}",
            config=self.config
        )
        return opt

    def optimize(self):
        self.get_path()

        for bandwidth in self.bandwidth_list:
            for num_devices in range(2, self.num_devices_max + 1):
                opt1 = self.optimize_once(bandwidth, num_devices, True, True)
                print(opt1.results)

def driver(config, threshold):
    opt_wrapper = OPT_WRAPPER(
        config=config,
        bandwidth_list=None,
        threshold=threshold,
    )
    opt_wrapper.optimize()

if __name__ == '__main__':

    threshold = 0.95
    print(f"Note: current threshold is {threshold}, "
          f"meaning that if increasing num_devices by one results in a change of speed up rate less than {1-threshold},"
          f" opt_num_devices won't be updated\n")

    for config in tqdm(OPT_WRAPPER.configs):
        driver(config, threshold)
