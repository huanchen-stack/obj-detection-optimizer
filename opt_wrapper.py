from optimizer import Optimizer
from simulatorv2 import Simulator
import os

config = "faster-nano"
path = os.path.abspath(os.getcwd())
path = os.path.join(path, f"testcases/{config}")

dep = os.path.join(path, "dep.csv")
prof = os.path.join(path, "prof.csv")
# part = os.path.join(path, "part.csv")
priority = os.path.join(path, "priority.csv")

bandwidth = 125 * float(input("Bandwidth: "))
iteration = int(input("Iteration: "))
prof_filenames = [prof] * int(input("Num Devices: "))

ignore_latency = False

# benchmark = 114.46748520400001  # agx
# benchmark = 194.446187  # agx no warm up
# benchmark = 274.2038  # agx time.time()
# benchmark = 47.999  # clarity32_new
# benchmark = 69.7  # clarity32
# benchmark = 139.144019769287  # nx
# benchmark = 331.0782  # cpu_vit

# benchmark = 0.509311  # faster-agx
benchmark = 1.905703  # faster-nano
# benchmark = 0.063555  # faster-clarity32


results = []
best = []
best_iter = []
r0 = []
r1 = []

opt0 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=True,
    dir=f"testcases/{config}",
)
results.append(opt0.report())

opt1 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=True,
    dir=f"testcases/{config}",
)
results.append(opt1.report())

opt2 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=False,
    reverse1=False,
    dir=f"testcases/{config}",
)
results.append(opt2.report())

opt3 = Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=iteration,
    benchmark=benchmark,
    reverse0=True,
    reverse1=False,
    dir=f"testcases/{config}",
)
results.append(opt3.report())

for result in results:
    best.append(result[0])
    best_iter.append(result[1])
    r0.append(result[2])
    r1.append(result[3])

Optimizer(
    dep_filename=dep,
    prof_filenames=prof_filenames,
    bandwidth=bandwidth,  # MBps
    ignore_latency=ignore_latency,
    iterations=best_iter[best.index(min(best))],
    benchmark=benchmark,
    reverse0=r0[best.index(min(best))],
    reverse1=r1[best.index(min(best))],
    dir=f"testcases/{config}",
)

print(f"\n\033[30;42m=========Result=========\033[0m")
print(f"Best result: {min(best)}")
print(f"Performance: {(benchmark - min(best)) / benchmark}")
print(f"Iteration: {best_iter[best.index(min(best))]}")
print(f"Setting: {r0[best.index(min(best))]}, {r1[best.index(min(best))]}")
