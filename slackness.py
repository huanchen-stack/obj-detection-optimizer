import os
import pandas as pd
from layer import Layer
from device import Device


class FindSlack(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=2000,
                 parallel=True,
                 ignore_latency=False,
                 iterations=1,
                 dir="",
                 ):
        super().__init__()
        self.bandwidth = bandwidth
        self.devices = {}
        self.device_names = []  # spinning through all devices
        self.layers = {}  # Dictionary of Layer objects: layername -> Layer objects
        self.parallel = parallel
        self.ignore_latency = ignore_latency
        self.iterations = iterations
        self.dir = dir

        self.results = []
        self.priorities = None  # file output

        # load and initialize devices
        parallel = True
        print(f"Device data-compute-parallel = {parallel}")

        self.num_devices = len(prof_filenames)
        self.device_names = [i for i in range(len(prof_filenames))]
        for name, prof_filename in zip(self.device_names, prof_filenames):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filenames[0])

        self.stack = []

        self.topological_sort()
        self.traverse()
        self.clean_up()
        self.topological_sort(reverse=True)
        self.traverse(reverse=True)
        self.find_slack()

    def load_dependencies(self, dep_filename):
        """
        Dependencies file has the following format for each line:
            source, destination, size (temporarily remove shape)
        Use source layer name as the name of the data
        Update Layer's dependencies and next lists
        """
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

    def topological_sort_helper(self, name, reverse):
        self.layers[name].explored = True
        pool = self.layers[name].next if not reverse else self.layers[name].dependencies
        for child in pool:
            if not self.layers[child].explored:
                self.topological_sort_helper(child, reverse)
        self.stack.append(name)

    def topological_sort(self, reverse=False):
        for name, layer in self.layers.items():
            if not layer.explored:
                self.topological_sort_helper(name, reverse)
        self.stack = self.stack[::-1]
        print(self.stack)
    
    def clean_up(self):
        for name, layer in self.layers.items():
            self.layers[name].explored = False
        self.stack.clear()

    def traverse(self, reverse=False):
        if not reverse:
            for name in self.stack:
                pool = [self.layers[dep].end_time for dep in self.layers[name].dependencies]
                Estart = max(pool) if pool else 0
                self.layers[name].pr_min = Estart
                self.layers[name].end_time = Estart + self.devices[0].time[name]
        else:
            for name in self.stack:
                pool = [self.layers[n].pr_max for n in self.layers[name].next]
                Lstart = min(pool) if pool else self.layers[name].pr_min
                Lstart -= self.devices[0].time[name]
                self.layers[name].pr_max = Lstart

    def find_slack(self):
        self.priorities = open(os.path.join(self.dir, "priority.csv"), "w")
        print(f"{'layer_name':<15} | {'Estart':<10} | {'Lstart':<10} | {'slack':<10} | {'priority':<10}")
        for name, layer in self.layers.items():
            slack = layer.pr_max - layer.pr_min
            priority = 1 / (slack + 0.01)
            print(f"{name:<15} | {round(layer.pr_min, 7):<10} | {round(layer.pr_max, 7):<10} | {round(slack, 7):<10} | {round(priority, 7):<10}")
            self.priorities.write(f"{name},{priority}\n")
        self.priorities.close()

            

import os
import sys

path = os.path.abspath(os.getcwd())
path = os.path.join(path, "testcases/yolov4/jetson_nx")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
part = os.path.join(path, "part.csv")
priority = os.path.join(path, "priority.csv")

# out = os.path.join(path, "out")
# sys.stdout = open(out, "w")

FindSlack(
    dep,
    [
        prof,
    ],
    dir="testcases/yolov4/jetson_nx"
)