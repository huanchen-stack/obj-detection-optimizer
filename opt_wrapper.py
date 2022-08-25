from queue import Empty
from random import expovariate
from optimizer import Optimizer
from simulatorv2 import Simulator
import os
from tqdm import tqdm


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
        'yolor-nano': 1.458861,
        'yolox-agx': 0.0916212,
        'yolox-nano': 1.76330,
    }
    
    bandwidths = {
        'agx': [
            750, 1000, 1250, 1500, 1750, 
            2000, 2250, 2500, 2750, 3000, float('inf'),
        ],
        'nano': [
            375, 500, 625, 750, 875, 
            1000, 1250, 1500, 1750, 2000, 3000, float('inf'),
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
        self.payload = []

    def get_path(self):
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"testcases/{self.config}")
        self.dep = os.path.join(path, "dep.csv")
        self.prof = os.path.join(path, "prof.csv")
        self.priority = os.path.join(path, "priority.csv")
        self.part = os.path.join(path, "part.csv")

    def run_simulatev2(self, bandwidth, num_devices):
        simv2 = Simulator(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * num_devices,
            bandwidth=bandwidth,
            priority_filename=self.priority,
            part_filename= self.part,
            ignore_latency=False,
        )
        return simv2.total_data_sent


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
            self.optimize_once(bandwidth, self.opt_num_devices[-1], best[2], best[3])
            payload = self.run_simulatev2(bandwidth, self.opt_num_devices[-1])
            self.payload.append(payload)

    def report(self):
        return {
            'bandwidths': OPT_WRAPPER.bandwidths[self.config.split('-')[1]],
            'opt_num_devices': self.opt_num_devices,
            'opt_speedup_rate': self.opt_speedup_rate,
            "payload": self.payload,
        }
        
def run(config, threshold):
    # config = input("config {model}-{device}: ")
    # threshold = float(input("threshold: "))
    opt_wrapper = OPT_WRAPPER(
        config=config,
        bandwidth_list=None,
        threshold=threshold,
    )
    opt_wrapper.optimize()
    res = opt_wrapper.report()

    # for key, value in res.items():
    #     print(key)
    #     print(value)

    # export
    with open(f'PLT_energy/{config}.csv', 'w') as f:
        f.write(f"bandwidth,optimizer,energy,device,payload\n")
        for i in range(len(res['bandwidths'])):
            f.write(f"{res['bandwidths'][i]},{res['opt_speedup_rate'][i]},0,{res['opt_num_devices'][i]},{res['payload'][i]}\n")

if __name__ == '__main__':
    # # customized input
    # config = input("config {model}-{device}: ")
    # threshold = float(input("threshold: "))
    # 'faster-agx', 'faster-nano', 'yolor-agx', 'yolor-nano',
    configs = ['yolox-agx', 'yolox-nano']
    threshold = 0.99
    print(f"Note: current threshold is {threshold}, meaning that if increasing num_devices by one \
results in a change of speed up rate less than {1-threshold}, opt_num_devices won't be updated\n")
    for config in tqdm(configs):
        run(config, threshold)