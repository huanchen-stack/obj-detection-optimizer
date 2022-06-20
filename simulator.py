from device import Device
import pandas as pd
from layer import Layer


class Simulator(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=200,
                 device_names=None,
                 priority_filename=None,
                 part_filename=None,
                 ignore_latency=False):
        super().__init__()
        self.bandwidth = bandwidth
        self.ignore_latency = ignore_latency

        self.current_device = 0  # spin
        self.device_names = []  # spinning through all devices
        self.devices = {}  # Dictionary of Device objects: device_name -> Device object
        self.layers = {}  # Dictionary of Layer objects: layername -> Layer objects
        self.priorities = {}  # Dictionary of integers: layername -> integer
        self.cut_points = {}

        self.time_result = {}
        self.mem_result = {}
        self.time_result_seg = {}

        self.stack = []  # for DFS
        self.waiting_queue = []  # for DFS: when layer cannot be explored due to
        #      dependencies are not fulfilled
        #      need to change device

        # load and initialize devices
        parallel = True
        print(f"Device parallel = {parallel}")
        if device_names is None:
            # TODO: should #device determined by prof?
            self.device_names = [str(i) for i in range(len(prof_filenames))]
        for name, prof_filename in zip(self.device_names, prof_filenames):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filenames)

        # if priority file is not given, init with even priorities
        if priority_filename is not None:
            self.load_priorities(priority_filename)
        else:
            for name in list(self.layers.keys()):
                self.priorities[name] = 1

        self.load_partitions(part_filename)  # Intermediate result of partition, now load from handcoded csv
        # self.partition(part_filename)

        print(self.device_names)
        for device in list(self.devices.values()):
            # TODO: Now exec has not much to do with assigned layers
            print(f"Device name: {device.name}, with layers: {device.assigned_layer}")
        print("{:<15} {:<15}".format("layer", "device"))
        for layer in list(self.layers.values()):
            print("{:<15} {:<15}".format(layer.name, layer.device_id))
        print(f"Layer priority: {self.priorities}")

        self.simulate()

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
        # TODO: Here size is with layers. If necessary, can be with dependencies.
        df_list = pd.read_csv(prof_filename).values.tolist()
        for layername, time, cpu, cuda, size, macs in df_list:
            self.layers[layername].size = size
            self.layers[layername].macs = macs

    def load_priorities(self, priority_filename):
        priorities = pd.read_csv(priority_filename).values.tolist()
        for layername, priority in priorities:
            self.priorities[layername] = priority

    def load_partitions(self, part_filename):
        partitions = pd.read_csv(part_filename).values.tolist()
        for layername, device_id in partitions:
            self.layers[layername].device_id = str(device_id)
            self.devices[str(device_id)].assigned_layer.append(layername)

    def go_through_path(self, layer_name, device_idx):
        # TODO: Perhaps only keep one of the following two lines
        if self.layers[layer_name].device_id is None:
            # if not assigned yet (see case 02)
            self.layers[layer_name].device_id = self.device_names[device_idx]
            self.devices[self.device_names[device_idx]].assigned_layer.append(layer_name)
        if layer_name == "output":
            return
        elif layer_name in self.cut_points.keys():
            for next_layer_name in self.cut_points[layer_name]:
                self.go_through_path(next_layer_name, device_idx + 1)
                device_idx += 1
            # go on with other next layers
            for next_layer_name in self.layers[layer_name].next:
                if next_layer_name not in self.cut_points[layer_name]:
                    self.go_through_path(next_layer_name, device_idx)
        else:
            for next_layer_name in self.layers[layer_name].next:
                self.go_through_path(next_layer_name, device_idx)

    def partition(self, part_filename):
        """
        Partition this graph. Assign every layer to a specific device.
        Assuming cut points are specified by two vertices.
        """
        if part_filename is None:
            # end to end on one device
            for key, value in self.layers.items():
                value.device_id = self.device_names[0]
        else:
            part_list = pd.read_csv(part_filename).values.tolist()
            for entry in part_list:
                if entry[0] not in self.cut_points.keys():
                    self.cut_points[entry[0]] = []
                self.cut_points[entry[0]].append(entry[1])
            self.go_through_path("input", 0)

    def device_exec(self, device_id, start_time, start_layer_name):
        """
        Update device current time.
        Returns the next layers.
        """
        device = self.devices[device_id]
        if start_layer_name not in device.assigned_layer:
            print(f"\033[31mLayer {start_layer_name} should not be executed on this device!\033[0m")
            exit(1)
        elif start_layer_name == "output":
            print(f"Device {device.name} generates output at time {start_time:.8f}")
            return
        else:
            if device.cur_time is not None:
                start_time = max(start_time, device.cur_time)
            print("")
            print(f"Device {device.name} is running: {start_layer_name}, starting at time {start_time:.8f}")
            cur_layer = self.layers[start_layer_name]
            num_needed_alien_data = 0
            for dep in cur_layer.dependencies:
                if not self.layers[dep].completed:
                    cur_layer.arrival_time_pool.append(start_time)
                    # cease exec
                    print(f"Dependencies NOT satisfied. Ceasing at {start_time:.8f} on device {device.name}")
                    print("")
                    return
                if self.layers[dep].device_id != device.name:
                    num_needed_alien_data += 1
            if len(cur_layer.arrival_time_pool) < num_needed_alien_data - 1:
                cur_layer.arrival_time_pool.append(start_time)
                print("Dependencies satisfied, but is missing data. Returning to previous device(s). ")
                return
            if len(cur_layer.arrival_time_pool) > 0:
                cur_layer.arrival_time_pool.append(start_time)
                device.cur_time = max(cur_layer.arrival_time_pool)
                print(f"Dependencies now satisfied. Arrival time pool: {cur_layer.arrival_time_pool}")
                print(f"Resuming at {device.cur_time:.8f} on device {device.name}")
            else:
                device.cur_time = start_time

            device.cur_time += device.time[start_layer_name]
            cur_layer.completed = True
            print(f"Finishing at time {device.cur_time:.8f}")

            if device_id not in self.time_result_seg.keys():
                self.time_result_seg[device_id] = 0
            self.time_result_seg[device_id] += device.time[start_layer_name]

            print(f"Next layers: {cur_layer.next}")
            cur_layer.next = sorted(cur_layer.next, key=lambda e: self.priorities[e], reverse=False)
            print(f"Sorted next layers: {cur_layer.next}")

            for next_layer_name in cur_layer.next:
                if self.layers[next_layer_name].completed:
                    # in case of:
                    #               1 -- 2 -- 3
                    #                \_______/
                    continue
                if next_layer_name == "output":
                    self.time_result[start_layer_name] = device.cur_time
                    continue
                if self.layers[next_layer_name].device_id != device_id:
                    transfer_latency = 0 if self.ignore_latency else cur_layer.size / self.bandwidth

                    print(f"Device {device.name} sends layer {cur_layer.name} output at time {device.cur_time: .8f} "
                          f"to device {self.layers[next_layer_name].device_id}, "
                          f"latency {transfer_latency:.8f}")

                    # change device
                    # for tail recursion
                    if not device.parallel:
                        self.time_result_seg[device_id] += transfer_latency
                        device.cur_time += transfer_latency
                        self.device_exec(self.layers[next_layer_name].device_id,
                                         device.cur_time,
                                         next_layer_name)
                    else:
                        self.device_exec(self.layers[next_layer_name].device_id,
                                         device.cur_time + transfer_latency,
                                         next_layer_name)
                    # if not device.parallel:
                    #     device.cur_time += transfer_latency
                else:
                    self.device_exec(device.name, device.cur_time, next_layer_name)

    def simulate(self):
        print(f"\n\033[30;44m=========Start Simulation=========\033[0m")
        self.device_exec("0", 0, "input")

        print(f"\n\033[30;42m=========Time Result=========\033[0m")
        print("{:<15} {:<15}".format("output_layer", "time (s)"))
        for key, value in self.time_result.items():
            print("{:<15} {:<15,.5f}".format(key, value))

        print(f"\n\033[30;42m=========Time Result per Device=========\033[0m")
        print("{:<15} {:<15}".format("device", "time (s)"))
        for key, value in self.time_result_seg.items():
            print("{:<15} {:<15,.5f}".format(key, value))

        print(f"\n\033[30;42m=========Mem Result=========\033[0m")
        print("{:<15} {:<15} {:<15} {:<15} {:<15}".format("device", "cpu sum (MB)", "cpu peak (MB)", "cuda sum (MB)", "cuda peak (MB)"))
        for name, device in self.devices.items():
            device.get_mem_consumption()

        print(f"\n\033[30;42m=========MACs Result=========\033[0m")
        print("{:<15} {:<15} {:<15}".format("device", "macs sum (M)", "macs peak (M)"))
        for name, device in self.devices.items():
            device.get_macs()
