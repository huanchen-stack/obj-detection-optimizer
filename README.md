# NS Optimizer

0. Per layer/block inference profile can be found in the [testcases](testcases) folder.

1. To find the optimized partition solutions and corresponding speed up rates, simply run `opt_wrapper.py` in the folder's root directory.
   - (Optional) Modify attributes and variables described in the following content.
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).

2. To analyze energy consumptions for drone communication under each *partition solution*, run `power-infer.py` in the folder's root directory.

3. To analyze energy consumption for computation (model inference), go to [this](https://github.com/huanchen-stack/tegraWATTS) repo

4. To generate plots, go to the `PLT_*` directories and run `PYPLT_*.py`.

---

# NS Optimizer Wrappers

[opt_wrapper.py](opt_wrapper.py) is an automation script that iterates throught all *configurations* (<*model*, *device*> pair, e.g. <yolo-v4, agx>), and brute force the optimal *number of drones* under different *bandwidth*. To run the optimization algrithm described in section V of the paper, [opt_wrapper.py](opt_wrapper.py) calls [optimizer.py](optimizer.py) for each configuration, number of drones, and bandwidth. 
   - [opt_wrapper_mem.py](opt_wrapper_mem.py) calls [optimizer_mem.py](optimizer_mem.py).
   - [opt_wrapper_battery.py](opt_wrapper_battery.py) calls [optimizer_battery.py](optimizer_battery.py).

The following guidelines illustrates how to use the wrappers on a neuro-network. 

### Inputs
Create a folder under [testcases](testcases) by the following format:
```shell
testcases/modelName-deviceCode/devicePowerMode/
```

To use an optimizer wrapper, the following files / attribute are needed in the corresponding folder:

#### 1. dep.csv:
This csv file contains the dependency relation between layers of a network. It has two columns: 
- Source 
- Destination
Each entry represents an edge in the network.

#### 2. prof.csv: 
This csv file contains the profiling result of every layer on a particular device. It has five columns:
- Layer name
- Average time consumption (in second)
- Output size (in MB)
- Average memory consumption (in MB)
- MACs (legacy column, set to zero)

### Default optimization 
While *optimizers* find one partitioning result for one model under one setting (a part.csv), *wrappers* generate multiple optimization results (opt.csv) in this folder. For default optimization:
- No device memory constrains. 
- Optimize by time.

please check the following attributes and variables.  

#### 1. Configs
The code block at the beginning of a wrapper specifies the model and device configurations. For example,
```python
configs = [
     'faster-nano',
     'faster-agx',
 ]
```
will run through these two configs and update the corresponding opt.csv in their directory.
#### 2. Benchmarks
Benchmarks are stored in 
```python
# benchmarks for optimization performance. Categorized by power mode. Unit: second
benchmarks = {
     "1": # device power mode
        {
         'faster-agx': 1.157999,
         'faster-nano': 2.686923,
        },
 }
```
#### 3. Bandwidth
The bandwidth of communication network between drones. Unit: mbps (will be converted to MBos later)
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
Config the device memory constrain (Unit: MB) in 
```python
memory_constrain = 1024*2 # MB
```
---

### Example

To run the memeory optimization for *yolo-v4* on *jetson-agx*:
  - Go to [optimizer_wrapper_mem.py](optimizer_wrapper_mem.py)
  - Locate *configurations* and comment out all other configurations except for `yolov4-agx`
  - Locate *benchmarks* and comment out all other configurations except for `yolov4-agx`
  - Change *memory_constraint* (in MegaBytes)
  - (Optional) Change Power Mode, Bandwidth range up to your purposes
  - Run `python3 optimizer_wrapper_mem.py`
  - Results can be found in the [data](data) directory

