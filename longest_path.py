import os
import pandas as pd
from layer import Layer
from device import Device


class FindLongestPath(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=2000,
                 parallel=True,
                 ignore_latency=True,
                 iterations=1,
                 dir="",
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

        self.results = []

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
        self.find_longest_path()

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

    def topological_sort_helper(self, name):
        self.layers[name].explored = True
        for child in self.layers[name].next:
            if not self.layers[child].explored:
                self.topological_sort_helper(child)
        self.stack.append(name)

    def topological_sort(self):
        for name, layer in self.layers.items():
            if not layer.explored:
                self.topological_sort_helper(name)
        self.stack = self.stack[::-1]
        print(self.stack)

    def find_longest_path(self):
        for name in self.stack:
            pool = [self.layers[dep].end_time for dep in self.layers[name].dependencies]
            dep_time = max(pool) if pool else 0
            self.layers[name].end_time = dep_time + self.devices[0].time[name]
            print(name, ":", self.layers[name].end_time)

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

FindLongestPath(
    dep,
    [
        prof,
    ],
)