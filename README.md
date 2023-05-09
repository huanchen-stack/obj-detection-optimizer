# NS Optimizer

1. To find the "optimal" partition solutions and corresponding speed up rates, run `opt_wrapper.py`.
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).

2. To analyze energy consumptions for each partition solution, run `power-infer.py`.

3. To generate plots, go to the `PLT_*` directories and run `PYPLT_*.py`.
---

# NS Optimizer Wrappers

The following guidelines illustrates how to use this code base on a neuro-network. 

### Inputs
Create a folder under [testcases](testcases) by the following format:
```shell
testcases/modelName-deviceCode/devicePowerMode/
```

To use an optimizer wrapper, the following files / attribute are needed in the corresponding this folder:

#### 1. dep.csv:
This csv file contains the dependency relation between layers of a network. It has two columns: 
- Source 
- Destination
Each entry represents an edge in the network.
#### 2. prof.csv: 
This csv file contains the profiling result of every layer on a particular device. It has five columns:
- Layer name
- Average time consumption (in second)
- Output size (in mb)
- Average memory consumption (in mb)
- MACs (legacy column, set to zero)

### Default optimization 
While optimizers find one partitioning result for one model under one setting (a part.csv), wrappers generate multiple optimization results (opt.csv) in this folder. For default optimization:
- No device memory constrains. 
- Optimize by time.

please check the following attributes and variables.  

#### 1. Configs
The code block at the beginning of a wrapper specifies the configurations to be run. For example,
```python
configs = [
     'faster-nano',
     'faster-agx',
 ]
```
will run through these two configs and update the corresponding opt.csv.
#### 2. Benchmarks
Benchmarks are stored in 
```python
# benchmarks for optimization performance. Categorized by power mode. Unit: second
benchmarks = {
     "1": {
         'faster-agx': 1.157999,
         'faster-nano': 2.686923,
     },
 }
```
#### 3. Bandwidth
The bandwidth of communication network between drones. Unit: mbps
* This can be set in bandwidth variable:
```python
bandwidths = {
     'agx':
         {'faster': [*range(900, 3400, 100)]},
     'nano':
         {'faster': [*range(900, 3400, 100)]},
 }
```
### Memory constrained scenario
#### 4. memory_constrain
Config the device memory constrain (Unit: mb) in 
```python
memory_constrain = 1024*2
```
---



