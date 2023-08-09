# 1. iterate through every potential partitioning points
# 2. Do simulation and record time
# 3. Find the best solution

import os
import shutil

import pandas as pd
from layer import Layer
from device import Device
from simulatorv2 import Simulator

# For now, please copy the configs from the corresponding wrapper

VERBOSE = False
POWER_MODE = 1

configs = [
    'faster-agx',
    # 'faster-nano',
    # 'yolor-agx',
    # 'yolor-nano',
    # 'yolox-agx',
    # 'yolox-nano',
    # 'yolov4-agx',
    # 'yolov4-nano',
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
    'agx':
        {'yolox': [*range(250, 2500, 250)],
         'yolor': [*range(250, 2500, 250)],
         # 'yolor': [250],
         'yolov4': [*range(250, 2500, 250)],
         # 'yolov4': [250],
         'faster': [*range(250, 2500, 250)],
         'yolos': [*range(250, 2500, 250)]},
    'nano':
        {'yolox': [*range(250, 2500, 250)],
         'yolor': [*range(250, 2500, 250)],
         'yolov4': [*range(250, 2500, 250)],
         'faster': [*range(250, 2500, 250)]},
}

class Optimizer(object):

    def __init__(self,
                 dep_filename,
                 prof_filename_edge,
                 prof_filename_cloud,
                 bandwidth=2000,  # mbps
                 parallel=True,
                 ignore_latency=False,  # * Whether to ignore transfer latency. Mainly for testing purpose.
                 iterations=1,
                 dir="",
                 benchmark=None,
                 ):
        super().__init__()
        self.dep_filename = dep_filename
        self.prof_filename_edge = prof_filename_edge
        self.prof_filename_cloud = prof_filename_cloud
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
        self.num_devices_list = []
        self.has_fixed = False
        self.opt_rate = None

        # load and initialize devices
        parallel = True  # whether a device can execute layers and send data to other devices at the same time

        self.num_devices = 1
        self.device_names = [0]
        for name, prof_filename in zip(self.device_names, prof_filename_edge):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filename_edge[0])

        self.optimize()

    def load_dependencies(self, dep_filename):
        """
        We load model inference dependencies into class objects, refer to:
            layer.py
            device.py
        for details.
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
        """
        We load model inference dependencies into class objects, refer to:
            layer.py
            device.py
        for details.
        """
        df_list = pd.read_csv(prof_filename).values.tolist()
        for layername, time, cpu, cuda, size, macs in df_list:
            self.layers[layername].size = size
            self.layers[layername].macs = macs

    def clean_up(self):
        """This function initializes block weights for each iteration."""
        for name, layer in self.layers.items():
            layer.end_time = 0
            layer.device_id = 0
        for name, device in self.devices.items():
            device.available_time = 0
            device.cur_time = 0

    def device_exec(self, cur_layer_name):
        """
        A recursive forward pass on the model.
            You can only proceed to a block/layer when all its dependencies are fulfilled,
            i.e., all its parent layers are already assigned to an available device.
        """

        if cur_layer_name == "output":  # the last block/layer of the model
            return
        else:
            cur_layer = self.layers[cur_layer_name]
            # Move the current layer to the cloud
            self.layers[cur_layer_name].device_id = 1

            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].device_id == 1:
                    # Completed in other iterations.
                    continue
                if next_layer_name == "output":  # the last block/layer of the model
                    self.layers["output"].device_id = 1
                    continue
                self.device_exec(next_layer_name)

    def optimize(self, write_csv=False):
        """
        The optimization includes two modules, optimize and backtrace, this is the first module.
        In this module, based on current priority/weight of the blocks, we assign different devices to them.
            the [core] part of optimize is done in the device_exec function
        Device assignment results, or, referred to as partition results, are stored in a csv file.
        """

        self.clean_up()

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        # Find a partition
        for layer in self.layers:
            # Find a layer that has not been cut and send its output to the cloud.
            # All the layers after this layer will be assigned to device 1.
            # After the assignment is finished, use a simulator to execute the partition, and record time & partition
            self.clean_up()

            for next_layer_name in self.layers[layer].next:
                # Assign device
                self.device_exec(next_layer_name)

            # create part file
            self.partitions = open("part.csv", "w")
            self.partitions.write(f"layername,device\n")
            for layer_for_csv in self.layers:
                self.partitions.write(f"{layer_for_csv},{self.layers[layer_for_csv].device_id}\n")
            self.partitions.close()

            # simulation
            sim = self.simulate(self.bandwidth)
            self.results.append({'layer': layer, 'time_result': sim.total_time, 'transfer_data_summary': sim.transfer_data_summary})

        # Find the best result
        self.results = sorted(self.results, key=lambda e: e['time_result'])
        best_time = self.results[0]['time_result']
        self.opt_rate = (benchmarks[str(POWER_MODE)][config] - best_time) / benchmarks[str(POWER_MODE)][config] * 100
        # print(f"Best result from NS-original: {best_time}")
        # print(f"Optimization: {self.opt_rate}%")
        # print("See part.csv for corresponding partition. ")
        self.clean_up()
        for next_layer_name in self.layers[self.results[0]['layer']].next:
            # Assign device
            # if next_layer_name == 'hd_conv13':
            #     a = 1
            self.device_exec(next_layer_name)
            # if next_layer_name == 'hd_conv13':
            #     for n, l in self.layers.items():
            #         print(f"{n}:\t{self.layers[n].device_id} \t{self.layers[n].end_time}")
            #     print("------")
        # recreate part file
        self.partitions = open(os.path.join("part.csv"), "w")
        self.partitions.write(f"layername,device\n")
        for layer_for_csv in self.layers:
            self.partitions.write(f"{layer_for_csv},{self.layers[layer_for_csv].device_id}\n")
        self.partitions.close()

        self.partitions = open(os.path.join(f"partitions/{config}/part-{int(bandwidth*8)}.csv"), "w")
        self.partitions.write(f"layername,device\n")
        for layer_for_csv in self.layers:
            self.partitions.write(f"{layer_for_csv},{self.layers[layer_for_csv].device_id}\n")
        self.partitions.close()
        sim = self.simulate(self.bandwidth)
        # for n, l in sim.layers.items():
        #     print(f"{n}:\t{sim.layers[n].device_id} \t{sim.layers[n].end_time}")
        # print("------")


    def report(self):
        # Find best result and the corresponding config
        # best = next(iter(self.results[0][1].values()))
        # r0 = "T" if self.reverse0 else "F"
        # r1 = "T" if self.reverse1 else "F"
        return [self.bandwidth, self.opt_rate]

    def simulate(self, bandwidth):
        """
        simulate calls the simulator class, defined in simulator_v2.py,
        which performs a forward pass of the inference by finding the longest path
        from the input to the output.
        """
        simv2 = Simulator(
            dep_filename=self.dep_filename,
            prof_filenames=['prof.csv', 'prof.csv'],
            bandwidth=bandwidth,
            priority_filename=os.path.join(path, 'priority.csv'),
            part_filename='part.csv',
            ignore_latency=False,
        )
        return simv2

if __name__ == '__main__':
    for config in configs:
        print(f"Working on {config}...")
        path = os.path.abspath(os.getcwd())
        path = os.path.join(path, f"../testcases/{config}/{POWER_MODE}")
        dep = os.path.join(path, "dep.csv")
        prof = os.path.join(path, "prof.csv")
        shutil.copyfile(prof, 'prof.csv')
        bandwidth_list = bandwidths[config.split('-')[1]][config.split('-')[0]]
        bandwidth_list = [bw * 0.125 for bw in bandwidth_list]
        res = {'bandwidths': [], 'opt_speedup_rate': [], 'payload': {}}
        for bandwidth in bandwidth_list:
            iterations = 1
            opt = Optimizer(
                dep_filename=dep,
                prof_filename_edge=[prof],
                prof_filename_cloud=[prof],
                bandwidth=bandwidth,  # mbs
                ignore_latency=False,
                iterations=iterations,
            )
            # print(opt.report())
            res['bandwidths'].append(opt.report()[0])
            res['opt_speedup_rate'].append(opt.report()[1])

        with open(f'data/{config}.csv', 'w') as f:
            f.write(f"bandwidth,optimizer,energy,device,payload\n")
            for i in range(len(res['bandwidths'])):
                f.write(
                    f"{res['bandwidths'][i] * 8},{res['opt_speedup_rate'][i]},0,2,{res['payload']}\n")

        print(f"{config} results stored.")
        print(f"==========")
