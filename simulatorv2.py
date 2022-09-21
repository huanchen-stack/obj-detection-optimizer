from device import Device
import pandas as pd
from layer import Layer


class Simulator(object):

    def __init__(self,
                 dep_filename,
                 prof_filenames,
                 bandwidth=5,
                 device_names=None,
                 priority_filename=None,
                 part_filename=None,
                 ignore_latency=False):
        super().__init__()
        self.bandwidth = bandwidth

        self.ignore_latency = ignore_latency
        self.results = []
        self.total_data_sent = 0
        self.transfer_data_summary = {}

        self.current_device = 0  # spin
        self.device_names = []  # spinning through all devices
        self.devices = {}  # Dictionary of Device objects: device_name -> Device object
        self.layers = {}  # Dictionary of Layer objects: layername -> Layer objects
        self.priorities = {}  # Dictionary of integers: layername -> integer
        self.cut_points = {}

        self.time_result = {}
        self.total_time = 0

        self.critical_path_graph = {}

        # load and initialize devices
        parallel = True
        if device_names is None:
            # TODO: should #device determined by prof?
            self.device_names = [str(i) for i in range(len(prof_filenames))]
        for name, prof_filename in zip(self.device_names, prof_filenames):
            self.devices[name] = Device(name, prof_filename, parallel=parallel)

        # load dependencies and initialize all Layers
        self.load_dependencies(dep_filename)
        self.load_macs_size(prof_filename)

        # if priority file is not given, init with even priorities
        if priority_filename is not None:
            self.load_priorities(priority_filename)
        else:
            for name in list(self.layers.keys()):
                self.priorities[name] = 1

        self.load_partitions(part_filename)  # Intermediate result of partition, now load from handcoded csv

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
            self.layers[layername].pr_max = priority

    def load_partitions(self, part_filename):
        partitions = pd.read_csv(part_filename).values.tolist()
        for layername, device_id in partitions:
            self.layers[layername].device_id = str(device_id)
            self.devices[str(device_id)].assigned_layer.append(layername)

    def clean_up(self):
        self.total_data_sent = 0
        for _, layer in self.layers.items():
            layer.end_time = 0
            # layer.device_id = None
        for _, device in self.devices.items():
            device.available_time = 0
            device.cur_time = 0

    def device_exec(self, cur_layer_name):
        """
        Update device current time.
        Returns the next layers.
        """
        # if cur_layer_name == "output":
        #     self.critical_path_graph["output"] = {
        #         'layername': cur_layer_name,
        #         'device_id': -1
        #     }
        #     return
        # else:
        cur_layer = self.layers[cur_layer_name]
        for dep in cur_layer.dependencies:
            if not self.layers[dep].completed:
                return

        device = self.devices[str(cur_layer.device_id)]
        dependency_arrival_timepool = []
        for dep in cur_layer.dependencies:
            dep_layer = self.layers[dep]
            transfer_latency = 0
            if (not self.ignore_latency) and str(dep_layer.device_id) != device.name:
                self.total_data_sent += dep_layer.size
                if dep != 'input':
                    if dep not in self.transfer_data_summary:
                        self.transfer_data_summary[dep] = {'count': 0, 'size': dep_layer.size}
                    self.transfer_data_summary[dep]['count'] += 1
                    transfer_latency = dep_layer.size / self.bandwidth
            # print(f"Receiving layer {dep} data from device {dep_layer.device_id}, "
            #       f"starting at {dep_layer.end_time:.4f}, latency {transfer_latency}.")
            end_time = dep_layer.end_time + transfer_latency
            dependency_arrival_timepool.append((end_time, dep))

        device = self.devices[str(cur_layer.device_id)]
        if device.last_exec_layername is None:
            assert device.available_time == 0
            dependency_arrival_timepool.append((0, ''))
        else:
            dependency_arrival_timepool.append((device.available_time, device.last_exec_layername))

        # find critical path
        if max(dependency_arrival_timepool)[1]:
            self.critical_path_graph[cur_layer_name] = {
                'layername': max(dependency_arrival_timepool)[1],
                'device_id': self.layers[max(dependency_arrival_timepool)[1]].device_id
            }

        end_time = max(dependency_arrival_timepool)[0] + device.time[cur_layer_name]
        self.layers[cur_layer_name].end_time = end_time
        self.layers[cur_layer_name].completed = True

        self.devices[str(cur_layer.device_id)].available_time = end_time
        self.devices[str(cur_layer.device_id)].last_exec_layername = cur_layer_name

        cur_layer.next = sorted(cur_layer.next, key=lambda e: self.layers[e].pr_max, reverse=True)

        for next_layer_name in cur_layer.next:
            if self.layers[next_layer_name].completed:
                continue
            if next_layer_name == "output":
                self.time_result[cur_layer_name] = cur_layer.end_time
                self.critical_path_graph["output"] = {
                    "layername": cur_layer_name,
                    "device_id": self.layers[cur_layer_name].device_id
                }
                continue
            self.device_exec(next_layer_name)

    def simulate(self):
        self.clean_up()

        self.layers["input"].end_time = 0
        self.layers["input"].device_id = '0'

        self.device_exec("input")

        self.total_time = list(self.time_result.values())[0]

    def single_cast(self):
        for _, d in self.transfer_data_summary.items():
            d['size'] *= d['count']
            d['count'] = 1

    def find_critical_path(self):

        if "input" not in self.transfer_data_summary:
            self.transfer_data_summary["input"] = {
                "size": 0
            }

        critical_path = [
            {
                "layername": "output",
                "device_id": -1,
            }
        ]
        # inserting from end to back
        while critical_path[0]["layername"] in self.critical_path_graph:
            critical_path.insert(0, self.critical_path_graph[critical_path[0]["layername"]])

        # sanitize device assignment
        #   in case transfer data size is set to 0 
        # for node in critical_path:
        #     layername = node["layername"]
        #     if layername in self.transfer_data_summary and self.transfer_data_summary[layername]["size"] == 0:
        #         for childname in self.layers[layername].next:
        #             for node_ in critical_path:
        #                 if childname == node_["layername"]:
        #                     node_["device_id"] = node["device_id"]
        #                     break

        res = []
        tmp = {
            'type': 'exec',
            'content': [],
        }
        prev_node = critical_path[0]
        for node in critical_path:

            if node["layername"] == "output":
                res.append(tmp)
                continue

            if node["device_id"] != prev_node["device_id"]:
                res.append(tmp.copy())
                tmp = {
                    'type': "data",
                    'content': self.transfer_data_summary[prev_node['layername']]['size']
                }
                res.append(tmp.copy())
                tmp = {
                    'type': 'exec',
                    'content': [],
                }

            tmp["content"].append(node["layername"])
            prev_node = node

        return res