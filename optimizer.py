import os
import pandas as pd
from layer import Layer
from device import Device

VERBOSE = False


class Optimizer(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=2000,  # mbps
                 parallel=True,
                 ignore_latency=False,  # * Whether to ignore transfer latency. Mainly for testing purpose.
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

        # those are different heuristics, when doing optimizaton, we try all possible values
        self.reverse0 = reverse0
        self.reverse1 = reverse1

        self.results = []
        self.num_devices_list = []
        self.has_fixed = False

        # load and initialize devices
        parallel = True  # whether a device can execute layers and send data to other devices at the same time
        
        self.num_devices = len(prof_filenames)
        self.device_names = [i for i in range(len(prof_filenames))]
        for name, prof_filename in zip(self.device_names, prof_filenames):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filenames[0])

        # Mark the first iteration where layers have the same priorities.
        self.FIRST_RUN = True 

        if self.iterations == 0:
            # throughout the optimization, we give different blocks different weights/priorities
            #   weights are initialized before the first run, but are assigned dynamically afterwards
            
            self.priorities = open(os.path.join(self.dir, "priority.csv"), "w")
            self.priorities.write(f"layername,priority\n")
            self.backtrace(write_csv=True)
            self.priorities.close()
            self.partitions = open(os.path.join(self.dir, "part.csv"), "w")  # record the partitioning decisions
            self.partitions.write(f"layername,device\n")
            self.optimize(write_csv=True)
            self.partitions.close()
            best = min(self.results)
            best_iter = self.results.index(best)
        else:
            # throughout the optimization, we give different blocks different weights/priorities
            #   weights are initialized before the first run, but are assigned dynamically afterwards
            
            self.optimize()

        self.FIRST_RUN = False  # Mark the first iteration where layers have the same priorities.

        for i in range(self.iterations):
            if i == self.iterations - 1:
                # through out the optimization, we give different blocks different weights/priorities
                #   weights are initialized before the first run, but are assigned dynamically afterwards
                
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
            else:
                # through out the optimization, we give different blocks different weights/priorities
                #   weights are initialized before the first run, but are assigned dynamically afterwards
                
                self.backtrace()
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
            layer.device_id = None
        for name, device in self.devices.items():
            device.available_time = 0
            device.cur_time = 0

    def decide_one_layer(self, cur_layer_name):
        """
        We perform the device assignment in the forward module, or referred to as the forward pass.
        We greedily aim to make the finish time of the current layer as early as possible.
        """

        # min(max(max(end_time + transfer_time), device_clock) + execution_time)
        device_results = []

        # When picking devices, we need them to be available (not already working on other tasks)
        #   we sort the devices by the earliest available time.
        sorted_device_names = list(self.devices.keys())
        sorted_device_names = sorted(sorted_device_names, key=lambda e: self.devices[e].available_time)

        # [< Greedily pick the most suited device! >] (details covered in the paper)
        #   computation latencies must be included
        #   communication latencies must be considerred: if the flow of execution meets a device change,
        #       we must include communication latency
        for device_name in sorted_device_names:
            device = self.devices[device_name]
            dependency_arrival_timepool = []
            for dep_name in self.layers[cur_layer_name].dependencies:
                dep_layer = self.layers[dep_name]
                # Compute data transfer latency and record end time for the current layer
                transfer_latency = 0
                if (not self.ignore_latency) and dep_layer.device_id != device.name:
                    transfer_latency = dep_layer.size / self.bandwidth
                end_time = dep_layer.end_time + transfer_latency  # + device.time[cur_layer_name]
                dependency_arrival_timepool.append(end_time)
            dependency_arrival_timepool.append(device.available_time)  # + device.time[cur_layer_name])
            device_results.append(max(dependency_arrival_timepool) + device.time[cur_layer_name])

        # In a single forward module, we don't want to update the device assignment of a block for multiple times
        #   We update the device assignment multiple times by calling forward modules multiple times
        if self.layers[cur_layer_name].fixed is not None:
            decision = self.layers[self.layers[cur_layer_name].fixed].device_id
            min_value = device_results[sorted_device_names.index(decision)]
            self.layers[cur_layer_name].device_id = decision
        else:
            min_value = min(device_results)
            decision = sorted_device_names[device_results.index(min_value)]
            self.layers[cur_layer_name].device_id = decision

        self.layers[cur_layer_name].completed = True
        self.layers[cur_layer_name].end_time = min_value
        self.devices[decision].available_time = min_value

        # A single block (layer) may have multiple parents, we must wait all parents to be completed
        #   before we can proceed to this block
        # Now that a decision is made,
        #   a dependency block may or may not be on the current block's decision device,
        #   therefore the decision may be improved according to the dependencies that are not on the decision device.

        same_source_dep_time = []
        for dep_layer_name in self.layers[cur_layer_name].dependencies:
            if self.layers[dep_layer_name].device_id == decision:
                same_source_dep_time.append(self.layers[dep_layer_name].end_time)
        if same_source_dep_time:
            earliest_ready_time = max(same_source_dep_time)
            possible_opt_pool = {}
            curr_start_time = self.layers[cur_layer_name].end_time - self.devices[decision].time[cur_layer_name]
            for dep_layer_name in self.layers[cur_layer_name].dependencies:
                # Find the possible partitioning decision that can improve the end time of current layer
                #   by recording and fixing that dependency that is not on the current decision with the current layer
                #   i.e. they will be assigned to the same device during the next iteration
                if self.layers[dep_layer_name].device_id != decision:
                    possible_opt_pool[dep_layer_name] = (
                            curr_start_time - (earliest_ready_time + self.devices[decision].time[dep_layer_name]))
            if possible_opt_pool and max(possible_opt_pool.values()) > 0:
                # Record the fixed layer suggestion for next iteration
                can_opt_dep_name = max(possible_opt_pool, key=possible_opt_pool.get)
                self.layers[can_opt_dep_name].fixed = self.layers[can_opt_dep_name].dependencies[0]
                self.layers[cur_layer_name].fixed = can_opt_dep_name

        return decision

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
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:  # dependencies are not fulfilled
                    return
            
            decision = self.decide_one_layer(cur_layer_name)

            if self.FIRST_RUN:
                # For the first iteration, layers have the same priorities.
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e],
                                        reverse=self.reverse0)
            else:
                # Otherwise, dynamically assigned priorities are used.
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=self.reverse1)

            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    # Completed in other iterations.
                    continue
                if next_layer_name == "output":  # the last block/layer of the model
                    self.layers["output"].device_id = decision
                    self.results.append(cur_layer.end_time)
                    self.num_devices_list.append(self.find_num_device())
                    continue
        
                self.device_exec(next_layer_name)

    def find_num_device(self):
        num_device = 0
        for key, value in self.layers.items():
            if value.device_id is not None and value.device_id > num_device:
                num_device = value.device_id
        return num_device + 1

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
        self.device_exec("input")

        # Save the result
        for layer_name, layer in self.layers.items():
            if write_csv:
                self.partitions.write(f"{layer_name},{layer.device_id}\n")

    def forward(self):
        """This is a legacy function and you may ignore it."""
        self.clean_up()

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        queue = ["input"]
        while queue:
            cur_layer_name = queue.pop(0)
            cur_layer = self.layers[cur_layer_name]

            # Check if all dependencies are finished.
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    return

            decision = self.decide_one_layer(cur_layer_name)

            # For the first iteration, layers have the same priorities.
            if self.FIRST_RUN:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e], reverse=True)
            else:
                cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=True)

            # Move to the next layer
            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    # Completed in other iterations.
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    continue
                queue.append(next_layer_name)

    def backtrace(self, write_csv=False):
        """
        The optimization includes two modules, optimize and backtrace, this is the second module.
        In this module, based on the prior device assignments of blocks/layers, we try to reassign priority,
            the priority is reassigned in a manner to facilitate the following optimize module,
            after that, backtrace module will be called again to facilitate the next next optmize module
            the [core] algorithm of this backtrace function is described in the paper, based on the following principle:
               [< blocks/layers on/closer to the critical path of the inference execution should be more prioritized >]
        The reassigned priorities after each call of backtrace, are stored in a csv file.
        """
        
        # Set priority bounds
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
        for name, layer in self.layers.items():
            if write_csv:
                self.priorities.write(f"{name},{layer.pr_max}\n")

    def report(self):
        # Find best result and the corresponding config
        best = min(self.results)
        best_iter = self.results.index(best)
        best_num_device = self.num_devices_list[best_iter]
        # r0 = "T" if self.reverse0 else "F"
        # r1 = "T" if self.reverse1 else "F"
        return [best, best_iter, self.reverse0, self.reverse1, best_num_device]
