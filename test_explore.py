import os
import sys
from optimizer import Optimizer

path = os.path.abspath(os.getcwd())
path = os.path.join(path, "testcases/explore")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")

out = os.path.join(path, "out")
sys.stdout = open(out, "w")

Optimizer(
    dep_filename=dep,
    prof_filenames=[
        prof,
        prof,
        prof,
    ],
    bandwidth=1,
    ignore_latency=False,
    iterations=5,
    dir="testcases/explore",
)