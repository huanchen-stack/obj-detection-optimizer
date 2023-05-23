import os
import sys

import pandas as pd
from layer import Layer
from device import Device

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
                 benchmark=None,
                 reverse0=True,
                 reverse1=True,
                 memory_constrain=None
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
        self.reverse0 = reverse0
        self.reverse1 = reverse1
        self.memory_constrain = memory_constrain
        self.success = True

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

        self.FIRST_RUN = True

        if self.iterations == 0:
            self.priorities = open(os.path.join(self.dir, "priority.csv"), "w")
            self.priorities.write(f"layername,priority\n")
            self.backtrace(write_csv=True)
            self.priorities.close()
            self.partitions = open(os.path.join(self.dir, "part.csv"), "w")
            self.partitions.write(f"layername,device\n")
            success = self.optimize(write_csv=True)
            if success:
                self.partitions.close()
                best = min(self.results)
                best_iter = self.results.index(best)
            else:
                print("Fail to optimize for partition CSV")
        else:
            success = self.optimize()
            if not success:
                self.success = False
            #     return success
            # self.forward()
        self.FIRST_RUN = False

        for i in range(self.iterations):
            if i == self.iterations - 1:
                self.priorities = open(os.path.join(self.dir, "priority.csv"), "w")
                self.priorities.write(f"layername,priority\n")
                self.backtrace(write_csv=True)
                self.priorities.close()
                self.partitions = open(os.path.join(self.dir, "part.csv"), "w")
                self.partitions.write(f"layername,device\n")
                success = self.optimize(write_csv=True)
                self.partitions.close()
                if not success:
                    self.success = False
                else:
                    best = min(self.results)
                    best_iter = self.results.index(best)
            else:
                self.backtrace()
                success = self.optimize()
                if not success:
                    self.success = False
                #     return success
                # self.forward()

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

    def clean_up(self):
        for name, layer in self.layers.items():
            layer.end_time = 0
            layer.device_id = None
        for name, device in self.devices.items():
            device.available_time = 0
            device.cur_time = 0
            device.assigned_layer.clear()

    def decide_one_layer(self, cur_layer_name):

        # min(max(max(end_time + transfer_time), device_clock) + execution_time)
        device_results = []

        sorted_device_names = list(self.devices.keys())
        sorted_device_names = sorted(sorted_device_names, key=lambda e: self.devices[e].available_time)

        def determine(x):
            # Check if memory constrain is exceeded.
            device = self.devices[x]
            if device.cuda_mem[cur_layer_name] > self.memory_constrain:
                print("!!!!!!!!!!!!!!!!!!! WARNING INSUFFICIENT MEMORY FOR SINGLE LAYER !!!!!!!!!!!!!!!!!!!")
                print("ABORTING")
                sys.exit(-1)
            return device.current_cuda_mem() + device.cuda_mem[cur_layer_name] < self.memory_constrain

        filtered_list = [x for x in sorted_device_names if determine(x)]
        if len(filtered_list) == 0:
            return -1

        for device_name in filtered_list:
            device = self.devices[device_name]
            dependency_arrival_timepool = []
            for dep_name in self.layers[cur_layer_name].dependencies:
                dep_layer = self.layers[dep_name]
                transfer_latency = 0
                if (not self.ignore_latency) and dep_layer.device_id != device.name:
                    transfer_latency = dep_layer.size / self.bandwidth
                end_time = dep_layer.end_time + transfer_latency  # + device.time[cur_layer_name]
                dependency_arrival_timepool.append(end_time)
            dependency_arrival_timepool.append(device.available_time)  # + device.time[cur_layer_name])
            device_results.append(max(dependency_arrival_timepool) + device.time[cur_layer_name])

        if self.layers[cur_layer_name].fixed is not None \
                and self.layers[self.layers[cur_layer_name].fixed].device_id in filtered_list:
            decision = self.layers[self.layers[cur_layer_name].fixed].device_id
            min_value = device_results[filtered_list.index(decision)]
            self.layers[cur_layer_name].device_id = decision
        else:
            min_value = min(device_results)
            decision = filtered_list[device_results.index(min_value)]
            self.layers[cur_layer_name].device_id = decision

        self.layers[cur_layer_name].completed = True
        self.layers[cur_layer_name].end_time = min_value
        self.devices[decision].available_time = min_value

        same_source_dep_time = []
        for dep_layer_name in self.layers[cur_layer_name].dependencies:
            if self.layers[dep_layer_name].device_id == decision:
                same_source_dep_time.append(self.layers[dep_layer_name].end_time)
        if same_source_dep_time:
            earliest_ready_time = max(same_source_dep_time)
            possible_opt_pool = {}
            curr_start_time = self.layers[cur_layer_name].end_time - self.devices[decision].time[cur_layer_name]
            for dep_layer_name in self.layers[cur_layer_name].dependencies:
                if self.layers[dep_layer_name].device_id != decision:
                    possible_opt_pool[dep_layer_name] = (
                            curr_start_time - (earliest_ready_time + self.devices[decision].time[dep_layer_name]))
            if possible_opt_pool and max(possible_opt_pool.values()) > 0:
                can_opt_dep_name = max(possible_opt_pool, key=possible_opt_pool.get)
                if determine(self.layers[self.layers[can_opt_dep_name].dependencies[0]].device_id):
                    self.layers[can_opt_dep_name].fixed = self.layers[can_opt_dep_name].dependencies[0]
                    self.layers[cur_layer_name].fixed = can_opt_dep_name

        # self.partitions.write(f"{cur_layer_name},{decision}\n")
        return decision

    def device_exec(self, cur_layer_name):
        """
        Update device current time.
        Returns the next layers.
        """
        if cur_layer_name == "output":
            return True
        else:
            cur_layer = self.layers[cur_layer_name]
            # if cur_layer_name == "add__0":
            #     for dep in cur_layer.dependencies:
            #         print(f"{dep}, {self.layers[dep].completed}")
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    return True

            decision = self.decide_one_layer(cur_layer_name)
            if decision == -1:
                return False
            self.devices[decision].assigned_layer.append(cur_layer_name)

            if self.FIRST_RUN:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e],
                                        reverse=self.reverse0)
            else:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=self.reverse1)

            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    self.results.append(cur_layer.end_time)
                    self.num_devices_list.append(self.find_num_device())
                    continue
                success = self.device_exec(next_layer_name)
                if not success:
                    return success
            return True

    def optimize(self, write_csv=False):
        self.clean_up()
        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0
        success = self.device_exec("input")
        if not success:
            return success
        for layer_name, layer in self.layers.items():
            if write_csv:
                self.partitions.write(f"{layer_name},{layer.device_id}\n")
        return success

    def find_num_device(self):
        num_device = 0
        for key, value in self.layers.items():
            if value.device_id is not None and value.device_id > num_device:
                num_device = value.device_id
        return num_device + 1

    def forward(self):

        self.clean_up()

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        queue = ["input"]
        while queue:
            cur_layer_name = queue.pop(0)
            cur_layer = self.layers[cur_layer_name]

            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    return

            decision = self.decide_one_layer(cur_layer_name)
            if decision == -1:
                return False
            self.devices[decision].assigned_layer.append(cur_layer_name)

            if self.FIRST_RUN:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e], reverse=True)
            else:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=True)

            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    continue
                queue.append(next_layer_name)
        return True

    def backtrace(self, write_csv=False):
        self.layers["output"].pr_max = 1000
        self.layers["output"].pr_min = 0
        queue = ["output"]
        while queue:
            cur_layer_name = queue.pop(0)
            cur_layer = self.layers[cur_layer_name]
            cur_layer.completed = False
            sorted_dep_layer_names = sorted(cur_layer.dependencies, key=lambda e: self.layers[e].end_time + (
                self.layers[e].size / self.bandwidth if self.layers[e].device_id != cur_layer.device_id else 0))
            for dep_layer_name in cur_layer.dependencies:
                if dep_layer_name == "input":
                    continue
                i = sorted_dep_layer_names.index(dep_layer_name)
                pr_max_ = cur_layer.pr_min + (i + 1) / len(cur_layer.dependencies) * (
                        cur_layer.pr_max - cur_layer.pr_min)
                pr_min_ = cur_layer.pr_min + i / len(cur_layer.dependencies) * (cur_layer.pr_max - cur_layer.pr_min)
                if (not self.layers[dep_layer_name].pr_max) or self.layers[dep_layer_name].pr_max < pr_max_:
                    self.layers[dep_layer_name].pr_max = pr_max_
                    self.layers[dep_layer_name].pr_min = pr_min_
                if dep_layer_name not in queue:
                    queue.append(dep_layer_name)

        self.layers["input"].completed = False
        self.layers["input"].pr_max = 0
        # for name, device in self.devices.items():
        #     device.available_time = 0

        for name, layer in self.layers.items():
            if write_csv:
                self.priorities.write(f"{name},{layer.pr_max}\n")

    def report(self):
        best = min(self.results)
        best_iter = self.results.index(best)
        best_num_device = self.num_devices_list[best_iter]
        # r0 = "T" if self.reverse0 else "F"
        # r1 = "T" if self.reverse1 else "F"
        return [best, best_iter, self.reverse0, self.reverse1, best_num_device]
