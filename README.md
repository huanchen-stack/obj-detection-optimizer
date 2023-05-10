# Obj-Detection-Optimizer

### How to use

0. Per layer/block inference profile can be found in the ****testcases**** folder

1. To find our "optimal" partition solutions and corresponding speed up rates, run `opt_wrapper.py`

2. To get energy consumptions for each partition, run `power-infer.py`

3. To complete plots xxx ~ xxx, go to the `PLT_energy` folder and run `PYPLT_energy.py`

#### To understand how our Optimizer works, read the following instructions before running the code.

### 1. test.py

```python
from optimizer import Optimizer
Optimizer(
    dep_filename="dep.csv",
    prof_filenames=[
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
        "prof.csv",
    ],
    bandwidth=2000,
    parallel=True,
    ignore_latency=True,
    iterations=1,
    dir="testcase/explore"
)
```

### 2. Attribute Explanation
#### 1. dep_filename
* This csv file contains the dependency relation between layers of a network. 
* It has two columns: source, destination. Every entry represents an edge in the network.
#### 2. prof_filenames
* This csv file contains the profiling result of every layer on a particular device.
* It has five columns: layer_name, time,cpu_mem, cuda_mem, size, MACs
#### 3. bandwidth
* The bandwidth of communication network between drones.
#### 4. parallel
* data-computation parallel. 
#### 5. ignore_latency
* Whether to ignore transfer latency. Mainly for testing. 
