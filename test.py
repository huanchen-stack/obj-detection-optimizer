from optimizer import Optimizer

Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
    ],
    bandwidth=2,
    parallel=True,
    ignore_latency=True,
)