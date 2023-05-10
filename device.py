import pandas as pd
from layer import Layer


class Device(object):

    def __init__(self, name, prof_filename, parallel=True):
        super().__init__()
        self.name = name
        self.parallel = parallel
        self.profile = {}  # Dictionary of Dictionary: layername -> two/three entries
        self.time = {}  # Dictionary of Time: layername -> time
        self.cpu_mem = {}  # Dictionary of Mem (cpu)
        self.cuda_mem = {}
        self.macs = {}
        self.cache = set()  # Set of cached data
        self.available_time = 0
        self.cur_time = 0
        self.assigned_layer = []
        self.load_profile(prof_filename)

    def load_profile(self, prof_filename):
        prof_df_list = pd.read_csv(prof_filename).values.tolist()
        for layername, time, cpu_mem, cuda_mem, size, macs in prof_df_list:
            self.time[layername] = time
            self.cpu_mem[layername] = cpu_mem
            self.cuda_mem[layername] = cuda_mem
            self.macs[layername] = macs

    def current_cuda_mem(self):
        cuda_sum = 0
        for layername in self.assigned_layer:
            cuda_sum += self.cuda_mem[layername]
        return cuda_sum

    def get_mem_consumption(self):
        cpu_peak, cpu_sum = 0.0, 0.0
        cuda_peak, cuda_sum = 0.0, 0.0
        for layername in self.assigned_layer:
            cpu_sum += self.cpu_mem[layername]
            cuda_sum += self.cuda_mem[layername]
            cpu_peak = cpu_peak if self.cpu_mem[layername] < cpu_peak else self.cpu_mem[layername]
            cuda_peak = cuda_peak if self.cuda_mem[layername] < cuda_peak else self.cuda_mem[layername]
        print("{:<15} {:<15,.4f} {:<15} {:<15,.2f} {:<15}".format(self.name, cpu_sum, cpu_peak, cuda_sum, cuda_peak))

    def get_macs(self):
        macs_peak, macs_sum = 0.0, 0.0
        for layername in self.assigned_layer:
            macs_sum += self.macs[layername]
            macs_peak = macs_peak if self.macs[layername] < macs_peak else self.macs[layername]
        print("{:<15} {:<15,.2f} {:<15}".format(self.name, macs_sum, macs_peak))
