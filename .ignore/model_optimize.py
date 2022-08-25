import os
import sys
from optimizer_ori import Optimizer

config = "faster-agx"

path = os.path.abspath(os.getcwd())
path = os.path.join(path, f"testcases/{config}")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
# part = os.path.join(path, "part.csv")
priority = os.path.join(path, "priority.csv")

# out = os.path.join(path, "out")
# sys.stdout = open(out, "w")

iterations = int(input("Iterations: "))
bandwidth = int(input("Bandwidth: "))
num_devices = int(input("Number of devices: "))

Optimizer(
    dep_filename=dep,
    prof_filenames = [prof] * num_devices,
    priority_filename=priority,
    bandwidth=bandwidth * 125,
    ignore_latency=False,
    iterations=iterations,
    dir=f"testcases/{config}"
)