from optimizer import Optimizer
from simulatorv2 import Simulator

bandwidth = 1000
ignore_latency = False
iteration = 5
# benchmark = 114.46748520400001  # agx
benchmark = 69.7  # clarity32

results = []
best = []
best_iter = []
r0 = []
r1 = []

opt0 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ],
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=True,
)
results.append(opt0.report())

opt1 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ],
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=True,
)
results.append(opt1.report())

opt2 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ],
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=False,
)
results.append(opt2.report())

opt3 = Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ],
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=False,
)
results.append(opt3.report())

for result in results:
    best.append(result[0])
    best_iter.append(result[1])
    r0.append(result[2])
    r1.append(result[3])

Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        # "prof.csv",
        # "prof.csv",
        # "prof.csv",
    ],
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=best_iter[best.index(min(best))],
    benchmark=benchmark,
    reverse0=r0[best.index(min(best))],
    reverse1=r1[best.index(min(best))],
)

print(f"\n\033[30;42m=========Result=========\033[0m")
print(f"Best result: {min(best)}")
print(f"Performance: {(benchmark - min(best)) / benchmark}")
print(f"Iteration: {best_iter[best.index(min(best))]}")
print(f"Setting: {r0[best.index(min(best))]}, {r1[best.index(min(best))]}")
