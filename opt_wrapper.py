from queue import Empty
from random import expovariate
import re
from optimizer import Optimizer
from simulatorv2 import Simulator
import os
from tqdm import tqdm

class OPT_WRAPPER(object):

    configs = [
        # 'faster-agx',
        'faster-nano',
        # 'yolor-agx',
        'yolor-nano',
        # 'yolox-agx',
        'yolox-nano',
        # 'yolov4-agx',
        'yolov4-nano'
    ]
    benchmarks = {
        'faster-agx': 0.509311,
        'faster-nano': 1.905703,
        'faster-clarity32': 0.063555,
        'yolor-agx': 0.1736289,
        'yolor-nano': 1.458861,
        'yolox-agx': 0.0916212,
        'yolox-nano': 1.76330,
        'yolov4-agx': 0.274311065,
        'yolov4-nano': 0.91332531,
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
            {'yolox': [*range(900, 3400, 100)],
             'yolor': [*range(900, 3400, 100)],
             'yolov4': [*range(900, 3400, 100)],
             'faster': [*range(900, 3400, 100)]},
        # 'nano': [*range(375, 1500, 125)],  # good graph
        'nano':
            {'yolox': [*range(250, 4500, 150)],
             'yolor': [*range(250, 4500, 150)],
             'yolov4': [*range(250, 8000, 250)],
             'faster': [*range(400, 2500, 100)]},
    }

    def __init__(self, config, bandwidth_list=None, threshold=0.99):
        super().__init__()
        self.config = config
        self.benchmark = OPT_WRAPPER.benchmarks[self.config]

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

    def simulate(self, bandwidth):
        simv2 = Simulator(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * self.num_devices_max,
            bandwidth=bandwidth,
            priority_filename=self.priority,
            part_filename=self.part,
            ignore_latency=False,
        )
        return simv2

    def optimize_once(self, bandwidth, num_devices, reverse0, reverse1, iterations=None):
        iterations = iterations if iterations is not None else self.iterations_default
        opt = Optimizer(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * num_devices,
            bandwidth=bandwidth,
            ignore_latency=False,  # always False
            iterations=iterations,
            benchmark=self.benchmark,
            reverse0=reverse0,
            reverse1=reverse1,
            dir=f"testcases/{self.config}",
        )
        return opt

    def optimize(self):
        self.get_path()

        for bandwidth in self.bandwidth_list:
            across_devices = []
            for num_devices in range(2, self.num_devices_max + 1):
                # try different optimization heuristics
                opt1 = self.optimize_once(bandwidth, num_devices, True, True)
                opt2 = self.optimize_once(bandwidth, num_devices, True, False)
                opt3 = self.optimize_once(bandwidth, num_devices, False, True)
                opt4 = self.optimize_once(bandwidth, num_devices, False, False)
                
                results = [opt1, opt2, opt3, opt4]
                results = [opt.report() for opt in results]
                results = sorted(results, key=lambda e: e[0])
                t = results[0]
                t.insert(1, 100-100*t[0]/self.benchmark)
                across_devices.append(t)

            # find optimal num_devices
            best = min(across_devices, key=lambda e: e[0])
            for i in range(len(across_devices)):
                num_devices = i + 2
                # if (across_devices[i][0] - best[0]) / best[0] <= 1 - self.threshold:
                if across_devices[i][0] * self.threshold <= best[0]:
                    self.opt_num_devices.append(num_devices)
                    self.opt_speedup_rate.append(across_devices[i][1])
                    best = across_devices[i]
                    break

            # get partitions for simulation
            args = [bandwidth, self.opt_num_devices[-1], best[3], best[4], best[2]]
            self.args.append(args)
            self.optimize_once(*args)
            # simulate to get payload info
            simv2 = self.simulate(bandwidth)
            transfer_data_summary = simv2.transfer_data_summary
            transfer_data_summary_raw = str(transfer_data_summary).replace(',', '|')  # for panda df read
            self.payload.append(transfer_data_summary_raw)
            # print(self.config, bandwidth, simv2.total_data_sent)

        self.sanitize()

    def sanitize(self):
        for i in range(len(self.bandwidth_list)):
            if i == 0:
                continue
            if self.opt_speedup_rate[i] < self.opt_speedup_rate[i-1]:  # Optimizer made a bad decision
                # get partitions from prev results
                self.optimize_once(*self.args[i-1])
                # use prev partition results and cur bandwidth to calculate new_opt latency & rate
                simv2 = self.simulate(self.bandwidth_list[i])
                new_opt_latency = simv2.total_time
                new_opt_rate = 100 - 100 * new_opt_latency / self.benchmark
                # update to output list
                self.opt_num_devices[i] = self.opt_num_devices[i-1]
                self.opt_speedup_rate[i] = max(new_opt_rate, self.opt_speedup_rate[i-1])  # FIXME: debug and remove max
                self.payload[i] = self.payload[i-1]
                self.args[i] = self.args[i-1]

    def report(self):
        return {
            'bandwidths': OPT_WRAPPER.bandwidths[self.config.split('-')[1]][self.config.split('-')[0]],
            'opt_num_devices': self.opt_num_devices,
            'opt_speedup_rate': self.opt_speedup_rate,
            'payload': self.payload,
        }
        
def driver(config, threshold):

    opt_wrapper = OPT_WRAPPER(
        config=config,
        bandwidth_list=None,
        threshold=threshold,
    )
    opt_wrapper.optimize()
    res = opt_wrapper.report()

    with open(f'data/{config}.csv', 'w') as f:
        f.write(f"bandwidth,optimizer,energy,device,payload\n")
        for i in range(len(res['bandwidths'])):
            f.write(f"{res['bandwidths'][i]},{res['opt_speedup_rate'][i]},0,{res['opt_num_devices'][i]},{res['payload'][i]}\n")


if __name__ == '__main__':

    threshold = 0.95
    print(f"Note: current threshold is {threshold}, meaning that if increasing num_devices by one results in a change of speed up rate less than {1-threshold}, opt_num_devices won't be updated\n")

    for config in tqdm(OPT_WRAPPER.configs):
        driver(config, threshold)

    # driver('yolor-agx', threshold)