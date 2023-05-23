import os

import numpy
import pandas as pd
from layer import Layer
from device import Device
from power_infer_battery import EnergyInferer

# Here we use layer.completed as a flag to indicate whether a layer has been sorted
from simulatorv2 import Simulator

VERBOSE = False


class Optimizer(object):
    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=2000,  # MB/s
                 parallel=True,
                 ignore_latency=False,
                 iterations=1,
                 dir="",
                 benchmark_energy=None,
                 reverse0=True,
                 reverse1=True,
                 config=None
                 ):
        super().__init__()
        self.bandwidth = bandwidth
        self.devices = {}
        self.device_names = []  # spinning through all devices
        self.layers = {}  # Dictionary of Layer objects: layername -> Layer objects
        self.priorities = {}  # Dictionary of integers: layername -> integer
        self.parallel = parallel
        self.ignore_latency = ignore_latency
        self.iterations = iterations
        self.dir = dir
        self.benchmark_energy = benchmark_energy
        self.config = config
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"testcases/{self.config}")
        self.dep = os.path.join(path, "dep.csv")
        self.prof = os.path.join(path, "prof.csv")
        self.priority = os.path.join(path, "priority.csv")
        self.part = os.path.join(path, "part.csv")

        self.reverse0 = reverse0
        self.reverse1 = reverse1

        self.results = []
        self.num_devices_list = []
        self.has_fixed = False

        # load and initialize devices
        parallel = True
        # print(f"Device data-compute-parallel = {parallel}")

        self.num_devices = len(prof_filenames)
        self.device_names = [i for i in range(len(prof_filenames))]
        for name, prof_filename in zip(self.device_names, prof_filenames):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filenames[0])

        self.optimize()
        best = min(self.results)
        best_iter = self.results.index(best)

    def load_dependencies(self, dep_filename):
        df_list = pd.read_csv(dep_filename).values.tolist()
        for entry in df_list:
            src = entry[0]
            dst = entry[1]
            if src not in self.layers.keys():
                self.layers[src] = Layer(src)
            if dst not in self.layers.keys():
                self.layers[dst] = Layer(dst)
            self.layers[src].next.append(dst)
            self.layers[dst].dependencies.append(src)

    def load_macs_size(self, prof_filename):
        df_list = pd.read_csv(prof_filename).values.tolist()
        for layername, time, cpu, cuda, size, macs in df_list:
            self.layers[layername].size = size
            self.layers[layername].macs = macs

    def find_least_unused_output_size(self):
        res_layer_name = ""
        least_size_unused = numpy.Inf
        for layer_name, layer in self.layers.items():
            if not layer.completed and least_size_unused > layer.size and layer.size != 0:
                least_size_unused = layer.size
                res_layer_name = layer_name
        return res_layer_name

    def assign_successors(self, start_layer_name, device_id):
        cur_device_id = self.layers[start_layer_name].device_id
        queue = self.layers[start_layer_name].next
        while queue:
            cur_layer_name = queue.pop(0)
            self.layers[cur_layer_name].device_id = device_id
            for next_layer_name in self.layers[cur_layer_name].next:
                if self.layers[next_layer_name].device_id != cur_device_id:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = device_id
                    self.num_devices_list.append(self.find_num_device())
                    continue
                queue.append(next_layer_name)

    def simulate(self, bandwidthm, num_devices_max):
        simv2 = Simulator(
            dep_filename=self.dep,
            prof_filenames=[self.prof] * num_devices_max,
            bandwidth=self.bandwidth,
            priority_filename=self.priority,
            part_filename=self.part,
            ignore_latency=False,
        )
        return simv2

    def find_num_device(self):
        num_device = 0
        for key, value in self.layers.items():
            if value.device_id is not None and value.device_id > num_device:
                num_device = value.device_id
        return num_device + 1

    def optimize(self):
        # set all layers on one device
        for layer_name, layer in self.layers.items():
            layer.device_id = 0

        num_used_devices = 1
        while num_used_devices < self.num_devices:
            next_layer = self.find_least_unused_output_size()
            self.assign_successors(next_layer, num_used_devices)
            # write partitions to file
            self.partitions = open(os.path.join(self.dir, "part.csv"), "w")
            self.partitions.write(f"layername,device\n")
            for layer_name, layer in self.layers.items():
                self.partitions.write(f"{layer_name},{layer.device_id}\n")
            self.partitions.close()

            simv2 = self.simulate(self.bandwidth, num_used_devices+1)
            transfer_data_summary = simv2.transfer_data_summary
            transfer_data_summary_raw = str(transfer_data_summary).replace(',', '|')  # for panda df read

            with open(f'data/{self.config}.csv', 'w') as f:
                f.write(f"bandwidth,optimizer,energy,device,payload\n")
                f.write(
                    f"{self.bandwidth},{0.0},0,{num_used_devices},{transfer_data_summary_raw}\n")

            inferer = EnergyInferer(self.config, True)
            total_energy = inferer.predict_energy(self.bandwidth, transfer_data_summary)
            battery_life = (num_used_devices/(total_energy + self.benchmark_energy)) / (1 / self.benchmark_energy)
            if len(self.results) > 0 and battery_life < max(self.results):
                # undo change and exit
                for layer_name, layer in self.layers.items():
                    if layer.device_id == num_used_devices:
                        layer.device_id = num_used_devices -1
                break
            self.layers[next_layer].completed = True
            self.results.append(battery_life)
            num_used_devices += 1

    def report(self):
        best = max(self.results)
        best_iter = self.results.index(best)
        best_num_device = self.num_devices_list[best_iter]
        # r0 = "T" if self.reverse0 else "F"
        # r1 = "T" if self.reverse1 else "F"
        return [best, best_iter, self.reverse0, self.reverse1, best_num_device]
