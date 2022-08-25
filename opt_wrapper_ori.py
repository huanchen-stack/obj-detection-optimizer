from queue import Empty
from optimizer import Optimizer
from simulatorv2 import Simulator
import os


[
    250, 375, 500, 625, 750, 875, 
    1000, 1125, 1250, 1375, 1500, 1625, 1750, 1875,
    2000, 2125, 2250, 2375, 2500, 2625, 2750, 2875,
    3000, float('inf')
]

class OPT_WRAPPER(object):

    benchmarks = {
        'faster-agx': 0.509311,
        'faster-nano': 1.905703,
        'faster-clarity32': 0.063555,
        'yolor-agx': 0.1736289,
        'yolor-nano': 1.458861
    }
    
    bandwidths = {
        'agx': [
            750, 1000, 1250, 1500, 1750, 
            2000, 2250, 2500, 2750, 3000, float('inf'),
        ],
        'nano': [
            375, 500, 625, 750, 875, 
            1000, 1250, 1500, 1750, 2000, float('inf'),
        ]
    }

    def __init__(self, config, bandwidth_list=None, threshold=0.99):
        super().__init__()
        self.config = config
        self.benchmark = OPT_WRAPPER.benchmarks[self.config]

        if bandwidth_list is None:
            self.bandwidth_list = OPT_WRAPPER.bandwidths[config.split('-')[1]]
            self.bandwidth_list = [bw * 0.125 for bw in self.bandwidth_list]
        else:
            self.bandwidth_list = bandwidth_list

        self.iterations_default = 5
        self.num_devices_max = 7
        self.threshold = threshold

        self.results = []
        self.opt_num_devices = []
        self.opt_speedup_rate = []

    def get_path(self):
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"testcases/{self.config}")
        self.dep = os.path.join(path, "dep.csv")
        self.prof = os.path.join(path, "prof.csv")
        self.priority = os.path.join(path, "priority.csv")

    def optimize_once(self, bandwidth, num_devices, reverse0, reverse1):
        opt = Optimizer(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * num_devices,
            bandwidth=bandwidth,
            ignore_latency=False,
            iterations=self.iterations_default,
            benchmark=self.benchmark,
            reverse0=reverse0,
            reverse1=reverse1,
            dir=f"testcases/{self.config}",
        )
        self.results.append(opt.report())

    def optimize(self):
        self.get_path()

        for bandwidth in self.bandwidth_list:

            across_devices = []
            for num_devices in range(2, self.num_devices_max + 1):
                self.results.clear()
                # try different optimization heuristics
                self.optimize_once(bandwidth, num_devices, True, True)
                self.optimize_once(bandwidth, num_devices, True, False)
                self.optimize_once(bandwidth, num_devices, False, True)
                self.optimize_once(bandwidth, num_devices, False, False)

                self.results = sorted(self.results, key=lambda e: e[0])
                t = self.results[0]
                t.insert(1, 100-100*t[0]/self.benchmark)
                across_devices.append(t)

            # find optimal num_devices
            best = min(across_devices, key=lambda e: e[0])
            for i in range(len(across_devices)):
                if across_devices[i][0] * self.threshold <= best[0]:
                    self.opt_num_devices.append(i+2)
                    self.opt_speedup_rate.append(across_devices[i][1])
                    break

    def report(self):
        return {
            'bandwidths': OPT_WRAPPER.bandwidths[self.config.split('-')[1]],
            'opt_num_devices': self.opt_num_devices,
            'opt_speedup_rate': self.opt_speedup_rate,
        }
        
# opt_wrapper = OPT_WRAPPER(
#     config=input("config {model}-{device}: "),
#     bandwidth_list=None,
#     threshold=0.99,
# )
# print("Note: current threshold is 0.99, \
# meaning that if increasing num_devices by one\
#  results in a change of speed up \
# rate less than 0.01, opt_num_devices won't be updated\n")
# opt_wrapper.optimize()
# print(opt_wrapper.report())
# exit(1)

config = input("config {model}-{device}: ")

# find files
path = os.path.abspath(os.getcwd())
path = os.path.join(path, f"testcases/{config}")
dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
priority = os.path.join(path, "priority.csv")

# independent variables (need automation)
bandwidth = 0.125 * float(input("Bandwidth/[Mbps]: "))
iteration = int(input("Iteration: "))
prof_filenames = [prof] * int(input("Num Devices: "))
ignore_latency = False

configs = {
    'faster-agx': 0.509311,
    'faster-nano': 1.905703,
    'faster-clarity32': 0.063555,
    'yolor-agx': 0.1736289,
    'yolor-nano': 1.458861,
    'yolov4-agx': 1,
    'yolov4-nano': 1,
}
benchmark = configs[config]

results = []
best = []
best_iter = []
r0 = []
r1 = []

opt0 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=True,
    dir=f"testcases/{config}",
)
print(opt0.report())
results.append(opt0.report())

opt1 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=True,
    dir=f"testcases/{config}",
)
results.append(opt1.report())

opt2 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=False,
    dir=f"testcases/{config}",
)
results.append(opt2.report())

opt3 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=False,
    dir=f"testcases/{config}",
)
results.append(opt3.report())

for result in results:
    best.append(result[0])
    best_iter.append(result[1])
    r0.append(result[2])
    r1.append(result[3])

Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=best_iter[best.index(min(best))],
    benchmark=benchmark,
    reverse0=r0[best.index(min(best))],
    reverse1=r1[best.index(min(best))],
    dir=f"testcases/{config}",
)

print(f"\n\033[30;42m=========Result=========\033[0m")
print(f"Best result: {min(best)}")
print(f"Performance: {(benchmark - min(best)) / benchmark}")
print(f"Iteration: {best_iter[best.index(min(best))]}")
print(f"Setting: {r0[best.index(min(best))]}, {r1[best.index(min(best))]}")
