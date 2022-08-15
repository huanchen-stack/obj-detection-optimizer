import os
import pandas as pd
from layer import Layer
from device import Device


class Optimizer(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=2000,
                 parallel=True,
                 ignore_latency=False,
                 iterations=1,
                 dir="",
                 benchmark=None,
                 reverse0=True,
                 reverse1=True,
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

        self.results = []
        self.has_fixed = False

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

        self.FIRST_RUN = True
        self.optimize()
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
                self.optimize(write_csv=True)
                self.partitions.close()

                best = min(self.results)
                best_iter = self.results.index(best)
                print(f"Best result is achieved at iteration #{best_iter}")

                for layer in self.layers.values():
                    if layer.fixed is not None:
                        print(f"Fixed layers: {layer.name} - {layer.fixed}")

                if benchmark is not None:
                    print(f"Optimization performance: {(benchmark - best) / benchmark}")
                print(f"All results: {self.results}")
            else:
                self.backtrace()
                self.optimize()
                # self.forward()

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
            self.layers[layername].size = float(size)
            self.layers[layername].macs = float(macs)

    def clean_up(self):
        for name, layer in self.layers.items():
            layer.end_time = 0
            layer.device_id = None
        for name, device in self.devices.items():
            device.available_time = 0
            device.cur_time = 0

    def decide_one_layer(self, cur_layer_name):
        print(f"Begin analyzing layer {cur_layer_name}. ")

        # min(max(max(end_time + transfer_time), device_clock) + execution_time)
        device_results = []

        sorted_device_names = list(self.devices.keys())
        sorted_device_names = sorted(sorted_device_names, key=lambda e: self.devices[e].available_time)
        for device_name in sorted_device_names:
            device = self.devices[device_name]
            dependency_arrival_timepool = []
            for dep_name in self.layers[cur_layer_name].dependencies:
                dep_layer = self.layers[dep_name]
                transfer_latency = 0
                if (not self.ignore_latency) and dep_layer.device_id != device.name:
                    print(type(dep_layer.size))
                    print(type(self.bandwidth))
                    transfer_latency = dep_layer.size / self.bandwidth * 1000

                end_time = dep_layer.end_time + transfer_latency  # + device.time[cur_layer_name]
                dependency_arrival_timepool.append(end_time)
            dependency_arrival_timepool.append(device.available_time)  # + device.time[cur_layer_name])
            print(f"The arrival time pool of dependencies on device {device_name} is {dependency_arrival_timepool}")
            device_results.append(max(dependency_arrival_timepool) + device.time[cur_layer_name])
        print(f"==>>decision pool(clock time): {device_results}")

        if self.layers[cur_layer_name].fixed is not None:
            decision = self.layers[self.layers[cur_layer_name].fixed].device_id
            min_value = device_results[sorted_device_names.index(decision)]
            # min_value = device_results[decision]
            self.layers[cur_layer_name].device_id = decision
            # self.layers[cur_layer_name].fixed = None
        else:
            min_value = min(device_results)
            decision = sorted_device_names[device_results.index(min_value)]
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
                self.layers[can_opt_dep_name].fixed = self.layers[can_opt_dep_name].dependencies[0]
                self.layers[cur_layer_name].fixed = can_opt_dep_name

        print(f"Decision for layer {cur_layer_name}: executed on device {decision}, "
              f"start at {min_value - self.devices[decision].time[cur_layer_name]}, end time {min_value}\n")
        # self.partitions.write(f"{cur_layer_name},{decision}\n")
        return decision

    def device_exec(self, cur_layer_name):
        """
        Update device current time.
        Returns the next layers.
        """
        if cur_layer_name == "output":
            return
        else:
            cur_layer = self.layers[cur_layer_name]
            # if cur_layer_name == "add__0":
            #     for dep in cur_layer.dependencies:
            #         print(f"{dep}, {self.layers[dep].completed}")
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    print(f"Dependency for {cur_layer_name} not satisfied. \n")
                    return

            decision = self.decide_one_layer(cur_layer_name)

            if self.FIRST_RUN:
                print("Sorting criteria: device end time")
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e], reverse=self.reverse0)
            else:
                print("Sorting criteria: priorities")
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=self.reverse1)

            print(f"Sorted branches: {cur_layer.next}")
            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    self.results.append(cur_layer.end_time)
                    continue
                self.device_exec(next_layer_name)

    def optimize(self, write_csv=False):

        self.clean_up()

        print(f"\n\033[30;44m=========Optimizinginging=========\033[0m")

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        self.device_exec("input")

        print("\n================DEVICE ASSIGNMENT================")
        print("{:<15} {:<15}".format("layer name", "device"))
        for layer_name, layer in self.layers.items():
            print("{:<15} {:<15}".format(layer_name, layer.device_id))
            if write_csv:
                self.partitions.write(f"{layer_name},{layer.device_id}\n")
        print("===============================================\n")

    def forward(self):

        self.clean_up()

        print(f"===============================================")
        print(f"====================BFS MODE===================")
        print(f"===============================================")

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        queue = ["input"]
        while queue:
            print(f"Current queue: {queue}")
            cur_layer_name = queue.pop(0)
            cur_layer = self.layers[cur_layer_name]

            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    print(f"Dependency for {cur_layer_name} not satisfied. \n")
                    return

            decision = self.decide_one_layer(cur_layer_name)

            if self.FIRST_RUN:
                print("Sorting criteria: device end time")
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e], reverse=True)
            else:
                print("Sorting criteria: priorities")
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=True)

            print(f"Sorted branches: {cur_layer.next}")
            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    continue
                queue.append(next_layer_name)

    def backtrace(self, write_csv=False):
        print(f"\n\033[30;45m=========Backtracking=========\033[0m")
        self.layers["output"].pr_max = 1000
        self.layers["output"].pr_min = 0
        queue = ["output"]
        while queue:
            print(f"Current queue: {queue}")
            cur_layer_name = queue.pop(0)
            cur_layer = self.layers[cur_layer_name]
            cur_layer.completed = False
            sorted_dep_layer_names = sorted(cur_layer.dependencies, key=lambda e: self.layers[e].end_time + (
                self.layers[e].size / self.bandwidth if self.layers[e].device_id != cur_layer.device_id else 0))
            print(f"On layer {cur_layer_name}, its dependencies are: {cur_layer.dependencies} (sorted by end time). ")
            for dep_layer_name in cur_layer.dependencies:
                if dep_layer_name == "input":
                    print(f"Reaching an input layer. Skip this iteration.")
                    continue
                i = sorted_dep_layer_names.index(dep_layer_name)
                pr_max_ = cur_layer.pr_min + (i + 1) / len(cur_layer.dependencies) * (
                        cur_layer.pr_max - cur_layer.pr_min)
                pr_min_ = cur_layer.pr_min + i / len(cur_layer.dependencies) * (cur_layer.pr_max - cur_layer.pr_min)
                if (not self.layers[dep_layer_name].pr_max) or self.layers[dep_layer_name].pr_max < pr_max_:
                    self.layers[dep_layer_name].pr_max = pr_max_
                    self.layers[dep_layer_name].pr_min = pr_min_
                    print(f"Updating the priority of layer {dep_layer_name}: new priority: [{pr_min_}, {pr_max_}]. ")
                if dep_layer_name not in queue:
                    queue.append(dep_layer_name)
                    print(f"Adding {dep_layer_name} to the queue. ")
            print("")

        self.layers["input"].completed = False
        self.layers["input"].pr_max = 0
        # for name, device in self.devices.items():
        #     device.available_time = 0

        print("\n================PRIORITIES================")
        for name, layer in self.layers.items():
            print(
                f"Layer {name:<10} has priority range ({str(layer.pr_min):<8}, {str(layer.pr_max):<8}]\t (finishing at time {layer.end_time})")
            if write_csv:
                self.priorities.write(f"{name},{layer.pr_max}\n")
        print("==========================================\n")

    def report(self):
        best = min(self.results)
        best_iter = self.results.index(best)
        # r0 = "T" if self.reverse0 else "F"
        # r1 = "T" if self.reverse1 else "F"
        return best, best_iter, self.reverse0, self.reverse1

