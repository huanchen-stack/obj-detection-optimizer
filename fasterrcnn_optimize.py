import os
import sys
from optimizer import Optimizer

path = os.path.abspath(os.getcwd())
path = os.path.join(path, "testcases/fasterrcnn")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")

# out = os.path.join(path, "out")
# sys.stdout = open(out, "w")

Optimizer(
    dep_filename=dep,
    prof_filenames=[
        prof,
        prof,
        # prof,
        # prof,
        # prof,
        # prof,
        # prof,
        # prof,
    ],
    bandwidth=1000,
    ignore_latency=False,
    iterations=1,
    dir="testcases/fasterrcnn"
)