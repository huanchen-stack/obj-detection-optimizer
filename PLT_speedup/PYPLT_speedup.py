import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

configs = ["faster-agx", "faster-nano", "yolor-agx", "yolor-nano", "yolox-agx", "yolox-nano"]

for i in range(1,6,2):
    config = configs[i]
    path = os.path.abspath(os.getcwd())
    path = os.path.join(path, f"testcases/{config}")

    fn_opt = os.path.join(path, "opt.csv")
    with open(fn_opt, 'r') as f_opt:
        reader = csv.reader(f_opt)
        BW = []
        TI = []
        BM = None
        SP = []
        ND = []
        PL = []
        skip = True
        for bandwidth, exec_time, benchmark, speedup, num_devices, payload_agg in reader:
            if skip:
                skip = False
                continue
            if bandwidth == "INF":
                bandwidth = float('inf')
            else:
                print(bandwidth)
                bandwidth = float(bandwidth)
                if bandwidth > 8:
                    continue
            speedup = float(speedup)
            BW.append(bandwidth)
            TI.append(exec_time)
            BM = benchmark
            SP.append(speedup)
            ND.append(num_devices)
            PL.append(payload_agg)

        print(config)
        print(BW)
        BW.sort()
        print(SP)
        plt.plot(BW, SP, label=config)
    
plt.xlabel("Bandwidth/[Gbps]")
plt.xticks(np.arange(0,7,0.5))
plt.ylabel("Speed_up/[%]")
plt.yticks(np.arange(0,0.25,0.05))

plt.legend()
plt.show()