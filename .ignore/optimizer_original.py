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
                 ):
        super().__init__()
        self.bandwidth = bandwidth
        self.devices = {}
        self.device_names = []  # spinning through all devices
        self.layers = {}  # Dictionary of Layer objects: layername -> Layer objects
        self.priorities = {}  # Dictionary of integers: layername -> integer
        self.parallel = parallel
        self.ignore_latency = ignore_latency
        
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

        self.partitions = open("part.csv", "w")
        self.partitions.write(f"layername,device\n")
        self.optimize()
        self.partitions.close()

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

    def decide_one_layer(self, cur_layer_name):
        print(f"Begin analyzing layer {cur_layer_name}. ")

        # min(max(max(end_time + transfer_time), device_clock) + execution_time)
        device_results = []
        for device_name, device in self.devices.items():
            dependency_arrival_timepool = []
            for dep_name in self.layers[cur_layer_name].dependencies:
                dep_layer = self.layers[dep_name]
                transfer_latency = 0
                if (not self.ignore_latency) and dep_layer.device_id != device.name:
                    transfer_latency = dep_layer.size / self.bandwidth

                end_time = dep_layer.end_time + transfer_latency + device.time[cur_layer_name]
                dependency_arrival_timepool.append(end_time)
            dependency_arrival_timepool.append(device.available_time + device.time[cur_layer_name])
            print(f"The arrival time pool of dependencies on device {device_name} is {dependency_arrival_timepool}")
            device_results.append(max(dependency_arrival_timepool))
        print(f"==>>decision pool(clock time): {device_results}")
        min_value = min(device_results)
        decision = device_results.index(min_value)
        self.layers[cur_layer_name].end_time = min_value
        self.layers[cur_layer_name].completed = True
        self.layers[cur_layer_name].device_id = decision
        self.devices[decision].available_time = min_value
        print(f"Decision for layer {cur_layer_name}: executed on device {decision}, end time {min_value}\n")
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
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    return

            decision = self.decide_one_layer(cur_layer_name)

            cur_layer.next = sorted(cur_layer.next, key=lambda e: self.devices[decision].time[e], reverse=True)
            cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].priority, reverse=True)
            
            print(f"Sorted branches: {cur_layer.next}")
            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    continue
                if next_layer_name == "output":
                    self.layers["output"].device_id = decision
                    continue
                self.device_exec(next_layer_name)

    def optimize(self):
        print(f"\n\033[30;44m=========Optimizinginging=========\033[0m")

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = 0

        self.device_exec("input")
        
        print(f"\n\033[30;42m=========Result=========\033[0m")
        print("{:<15} {:<15}".format("layer name", "device"))
        for layer_name, layer in self.layers.items():
            print("{:<15} {:<15}".format(layer_name, layer.device_id))
            self.partitions.write(f"{layer_name},{layer.device_id}\n")

