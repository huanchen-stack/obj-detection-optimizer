import os
import sys
from optimizer import Optimizer

path = os.path.abspath(os.getcwd())
path = os.path.join(path, "testcases/fasterrcnn")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
# part = os.path.join(path, "part.csv")
priority = os.path.join(path, "priority.csv")

# out = os.path.join(path, "out")
# sys.stdout = open(out, "w")

Optimizer(
    dep_filename=dep,
    prof_filenames=[
        prof,
        prof,
        prof,
        # prof,
        # prof,
        # prof,
        # prof,
        # prof,
    ],
    priority_filename=priority,
    bandwidth=5000,
    ignore_latency=False,
    iterations=5,
    dir="testcases/fasterrcnn"
)