from queue import Empty
from random import expovariate
import re
from optimizer import Optimizer
from simulatorv2 import Simulator
import os
from tqdm import tqdm

# Device power mode while profiling, specifying the benchmarks
# POWER_MODE = "1" is the default, where CLOCK_RATE and POW_IN are fixed
#   Devices may have various power mode, check specification for details
POWER_MODE = "1"


class OPT_WRAPPER_MULTIDEVICE(object):
    # Input configs for the current execution
    #     the opt_wrapper will iterate through all configurations one by one
    # For the purpose of automation, you may uncomment all configurations
    configs = [
        # 'faster',
        # 'yolor',
        # 'yolox',
        'yolov4',
        # 'yolos-agx'
    ]

    # benchmarks for optimization performance. Categorized by power mode. Unit: second
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

    # Bandwidths that the optimizer will run through. Categorized by device model. Unit: mbps
    bandwidths = {
        'yolox': [*range(250, 4500, 150)],
        'yolor': [*range(250, 4500, 150)],
        'yolov4': [*range(250, 8000, 250)],
        'faster': [*range(900, 3400, 100)]
    }

    def __init__(self, config, bandwidth_list=None, threshold=0.99, devices=None):
        super().__init__()
        self.config = config
        self.benchmark = OPT_WRAPPER_MULTIDEVICE.benchmarks[POWER_MODE][self.config + '-' + devices[0]]
        self.devices = devices
        self.prof = []

        if bandwidth_list is None:
            self.bandwidth_list = OPT_WRAPPER_MULTIDEVICE.bandwidths[config]
        else:  # users may specify a bandwidth list instead of using existing ranges in the class
            self.bandwidth_list = bandwidth_list
        self.bandwidth_list = [bw * 0.125 for bw in self.bandwidth_list]  # turn to MBps

        self.iterations_default = 5
        self.num_devices_max = len(self.devices)  # opt_wrapper iterate all possible num_device values from (1, num_devices_max]
        self.threshold = threshold

        self.opt_num_devices = []
        self.opt_speedup_rate = []
        self.payload = []
        self.args = []

    def get_path(self):
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"../testcases/{self.config + '-' + self.devices[0]}/{POWER_MODE}")
        self.dep = os.path.join(path, "dep.csv")

        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"testcases/{self.config}/")
        self.priority = os.path.join(path, "priority.csv")
        self.part = os.path.join(path, "part.csv")
        for device in self.devices:
            path = os.path.abspath(os.getcwd())
            path = os.path.join(path, f"../testcases/{self.config + '-' + device}/{POWER_MODE}")
            self.prof.append(os.path.join(path, "prof.csv"))

    def simulate(self, bandwidth):
        """
        simulate calls the simulator class, defined in simulator_v2.py,
        which performs a forward pass of the inference by finding the longest path
        from the input to the output.
        """
        simv2 = Simulator(
            dep_filename=self.dep,
            prof_filenames=self.prof,
            bandwidth=bandwidth,
            priority_filename=self.priority,
            part_filename=self.part,
            ignore_latency=False,
        )
        return simv2

    def optimize_once(self, bandwidth, num_devices, reverse0, reverse1, iterations=None):
        """
        optimize_once calls the optimizer class, defined in optimizer.py,
        which performs a forward+backward optimization with the algorithm described in the paper.
        parameter: [iterations] is used in the optimizer class, see optimizer.py for details.
        """
        iterations = iterations if iterations is not None else self.iterations_default
        opt = Optimizer(
            dep_filename=self.dep,
            prof_filenames=self.prof,
            bandwidth=bandwidth,
            ignore_latency=False,
            iterations=iterations,
            benchmark=self.benchmark,
            reverse0=reverse0,
            reverse1=reverse1,
            dir=f"testcases/{self.config}",
        )
        return opt

    def optimize(self):
        """
        opt_wrapper iterates through all possible bandwidth in the list for analysis:
            opt_wrapper iterates through all possible number of devices and pick
            the most optimal one (usually the more devices you use, the more speedup
            you can get, but the more extra energy consumption would be incurred):
                for each of those iterations, four different heuristics are tried
                we pick the best of all heuristics
            if increasing the number of devices won't introduce a significant speedup
            the device increment would be discarded
        """
        self.get_path()

        for bandwidth in self.bandwidth_list:
            # opt_wrapper iterates through all possible bandwidth in the list for analysis:

            across_devices = []
            for num_devices in range(1, self.num_devices_max + 1):
                # opt_wrapper iterates through all possible number of devices and pick
                #   the most optimal one (usually the more devices you use, the more speedup
                #   you can get, but the more extra energy consumption would be incurred):

                # four different heuristics are tried, we pick the best of all heuristics
                opt1 = self.optimize_once(bandwidth, num_devices, True, True)
                opt2 = self.optimize_once(bandwidth, num_devices, True, False)
                opt3 = self.optimize_once(bandwidth, num_devices, False, True)
                opt4 = self.optimize_once(bandwidth, num_devices, False, False)

                opt_results = [opt1, opt2, opt3, opt4]
                # opt_results = [opt4]
                results = []
                for opt in opt_results:
                    report = opt.report()
                    if report is not None:
                        results.append(report)
                results = sorted(results, key=lambda e: e[0])
                t = results[0]
                t.insert(1, 100 - 100 * t[0] / self.benchmark)
                across_devices.append(t)

            # we only the best result from the four heuristics of our choice
            best = min(across_devices, key=lambda e: e[0])
            for i in range(len(across_devices)):
                num_devices = across_devices[i][5]
                # if (across_devices[i][0] - best[0]) / best[0] <= 1 - self.threshold:
                if across_devices[i][0] * self.threshold <= best[0]:
                    self.opt_num_devices.append(num_devices)
                    self.opt_speedup_rate.append(across_devices[i][1])
                    best = across_devices[i]
                    break

            # We need partition details and payload details for energy analysis:
            #   1. get partitions for simulation
            args = [bandwidth, self.opt_num_devices[-1], best[3], best[4], best[2]]
            self.args.append(args)
            self.optimize_once(*args)
            #   2. simulate to get payload info
            simv2 = self.simulate(bandwidth)
            transfer_data_summary = simv2.transfer_data_summary
            transfer_data_summary_raw = str(transfer_data_summary).replace(',', '|')  # for panda df read
            self.payload.append(transfer_data_summary_raw)

        # sanitize: find optimal num_devices
        #   if increasing the number of devices won't introduce a significant speedup
        #   the device increment would be discarded
        self.sanitize()

    def sanitize(self):
        """
        the more devices you use, the more speedup you can get,
            but the more extra energy consumption would be incurred
        if increasing the number of devices won't introduce a significant speedup (by threshold%)
            the device increment would be discarded
        """
        for i in range(len(self.bandwidth_list)):
            if i == 0:
                continue
            if self.opt_speedup_rate[i] < self.opt_speedup_rate[i - 1]:  # Optimizer made a bad decision
                self.optimize_once(*self.args[i - 1])  # get partitions from prev results

                # use prev partition results and cur bandwidth to calculate new_opt latency & rate
                simv2 = self.simulate(self.bandwidth_list[i])
                new_opt_latency = simv2.total_time
                new_opt_rate = 100 - 100 * new_opt_latency / self.benchmark

                # update to output list
                self.opt_num_devices[i] = self.opt_num_devices[i - 1]
                self.opt_speedup_rate[i] = max(new_opt_rate, self.opt_speedup_rate[i - 1])
                self.payload[i] = self.payload[i - 1]
                self.args[i] = self.args[i - 1]

    def report(self):
        return {
            'bandwidths': OPT_WRAPPER_MULTIDEVICE.bandwidths[self.config],
            'opt_num_devices': self.opt_num_devices,
            'opt_speedup_rate': self.opt_speedup_rate,
            'payload': self.payload,
        }


def driver(config, threshold):
    opt_wrapper = OPT_WRAPPER_MULTIDEVICE(
        config=config,
        bandwidth_list=None,
        threshold=threshold,
        devices=['agx','agx']
    )
    opt_wrapper.optimize()
    res = opt_wrapper.report()

    with open(f'data/{config}.csv', 'w') as f:
        f.write(f"bandwidth,optimizer,energy,device,payload\n")
        for i in range(len(res['bandwidths'])):
            f.write(
                f"{res['bandwidths'][i]},{res['opt_speedup_rate'][i]},0,{res['opt_num_devices'][i]},{res['payload'][i]}\n")


if __name__ == '__main__':

    threshold = 0.99
    print(f"Note: current threshold is {threshold}, "
          f"meaning that if increasing num_devices by one results in a change of speed up rate less than {1 - threshold},"
          f" opt_num_devices won't be updated\n")

    for config in tqdm(OPT_WRAPPER_MULTIDEVICE.configs):
        driver(config, threshold)
