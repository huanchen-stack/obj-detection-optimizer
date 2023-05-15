# NS Optimizer

A **NS optimizer** is a unit program that finds a partitioning strategy for a neuro-network model under some specified configurations. To find the best partitioning strategy across a range of data-transfer bandwidth for a model, an **optimizer wrapper** is needed to iterate through all scenarios.

1. To find the optimized partition solution and corresponding speed up rate under specified bandwidth, run [opt_wrapper.py](opt_wrapper.py) with bandwidth modified to preferred value. 
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).
   - For example, `'yolov4': [*range(250, 260, 250)],` specifies the bandwidth to 250 mbps.
   - Results are stored in `testcases/model-device/part.csv` and `testcases/model-device/priority.csv`
   - An NS Colorer can be used to visualize the results.  
2. To find the optimized partition solutions and corresponding speed up rates across a range of bandwidth, run [opt_wrapper.py](opt_wrapper.py) with bandwidth modified to preferred range.
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).
   - For example, `'yolov4': [*range(250, 1100, 250)],` specifies the bandwidth to 250, 500, 750, 1000 mbps.
   - The results will be stored in [data](data). Note that the .csv files in `testcases/model-device/` directories records only the results under the largest bandwidth. 
3. To analyze energy consumptions for each *partition solution* (recorded in [data](data)), run [power-infer.py](power-infer.py).
4. To generate plots, go to the `PLT_*` directories and run `PYPLT_*.py`.

---

# NS Optimizer Wrappers

The following guidelines illustrates how to use the wrappers on a neuro-network model. 

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

## Example: Yolov4
In this example, we will find the optimized partitioning strategies for Yolov4 across a range of data transfer bandwidth, and plot the battery life optimization graphs. We will use Jetson-AGX power mode 1 profiling data.

The input files can be found in [yolov4-agx/1](testcases%2Fyolov4-agx%2F1). 

### Default optimization
1. Configer [opt_wrapper.py](opt_wrapper.py) so that other models are omitted. 
```python
# Device power mode while profiling, specifying the benchmarks
POWER_MODE = "1"
# Input configs for the current execution
configs = [
     # 'faster-agx',
     # 'faster-nano',
     # 'yolor-agx',
     # 'yolor-nano',
     # 'yolox-agx',
     # 'yolox-nano',
     'yolov4-agx',
     # 'yolov4-nano'
]
```
   - For memory constrained scenario, use [opt_wrapper_mem.py](opt_wrapper_mem.py) with memory constrains (Unit: MB). (same below)
   - For battery life prioritized scenario, use [opt_wrapper_battery.py](opt_wrapper_battery.py). (same below)

2. Adjust the data transfer bandwidth range as needed. The following is just an example. 
```python
# Bandwidths that the optimizer will run through. Categorized by device model. Unit: mbps
bandwidths = {
  'agx':
      {
       'yolov4': [*range(250, 2100, 250)],
      }
}
```
3. run `python3 opt_wrapper.py`. Check [data/yolov4-agx.csv](data%2Fyolov4-agx.csv) for results. 

4. run `python3 power-infer.py`. 
5. run `python3 PLT_energy/PYPLT_energy.py`. Check [yolov4-agx.png](PLT_energy%2F1%2Fyolov4-agx.png) for battery life optimization plots. 
