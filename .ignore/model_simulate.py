import os
import sys
from simulatorv2 import Simulator

config = "yolor-nano"

path = os.path.abspath(os.getcwd())
path = os.path.join(path, f"testcases/{config}")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
part = os.path.join(path, "part.csv")
priority = os.path.join(path, "priority.csv")

# out = os.path.join(path, "out")
# sys.stdout = open(out, "w")

Simulator(
    dep,
    [
        prof,
        prof,
        prof,
        prof,
        prof,
        prof,
        prof,
    ],
    part_filename=part,
    priority_filename=priority,
    bandwidth = 125 * float(input("Bandwidth: ")),
)