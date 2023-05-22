# NS Optimizer
There are two types of NS optimization tools in this repo. An **NS optimizer** is a unit program that finds an optimized partitioning strategy for a neuro-network model under some specified configurations. To find the best partitioning strategies across a range of data-transfer bandwidth for a model, an **NS optimizer wrapper** is needed to iterate through all scenarios.
1. Per layer/block inference profile can be found in the [testcases](testcases) folder.
2. At the top of the wrappers, you can configure settings like models, bandwidth, devices, etc. for the experiment. 
3. We have data for different power modes for Jetson AGX and Jetson Nano boards. Under power mode 0, the devices are running at the full power; under power mode 1, Jetson AGX is running under 20W limit, and Jetson Nano is running under 5W limit. (See Jetson board handbook for mode info)
4. To find the optimized partition solution and corresponding speed up rate under **specified bandwidth**, run [opt_wrapper.py](opt_wrapper.py) with bandwidth modified to preferred value. 
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).
   - For example, `'yolov4': [1000],` specifies the bandwidth to 1000 mbps.
   - (Optional) Modify attributes and variables described in the following content.
   - Results are stored in `testcases/model-device/part.csv` and `testcases/model-device/priority.csv`
   - Optionally, the dep.csv, prof.csv, and part.csv files can be taken to the NS-DOT-Visualizer to visualize the results.  
5. To find the optimized partition solutions and corresponding speed up rates across **a range of bandwidth**, run [opt_wrapper.py](opt_wrapper.py) with bandwidth modified to preferred range.
   - For devices with memory constrains, use [opt_wrapper_mem.py](opt_wrapper_mem.py).
   - To optimize by battery life instead of execution time, use [opt_wrapper_battery.py](opt_wrapper_battery.py).
   - For example, `'yolov4': [*range(250, 1100, 250)],` specifies the bandwidth to 250, 500, 750, 1000 mbps.
   - (Optional) Modify attributes and variables described in the following content.
   - The results will be stored in [data](data). Note that the .csv files in `testcases/model-device/` directories records only the results under the largest bandwidth. 
6. To analyze energy consumptions for drone communication under each *partition solution* (recorded in [data](data)), run [power-infer.py](power-infer.py).
7. To analyze energy consumption for computation (model inference), go to [this](https://github.com/huanchen-stack/tegraWATTS) repo 
8. To generate plots, go to the `PLT_*` directories and run `PYPLT_*.py`.
9. For other plots in the paper, see [this](https://github.com/Yanmeeei/NS-DOT-visualizers) repo
## Prerequiste
Install `seaborn` package from pip
```bash
pip install seaborn
```
---
# NS Optimizer Wrappers
[opt_wrapper.py](opt_wrapper.py) is an automation script that iterates through a list of <*model*, *device*> pair, (e.g. <yolo-v4, agx>), and apply the partition algorithm given the *number of drones* under different *bandwidth*. To run the optimization algrithm described in section V of the paper, [opt_wrapper.py](opt_wrapper.py) calls [optimizer.py](optimizer.py) for each configuration, number of drones, and bandwidth. 
   - [opt_wrapper_mem.py](opt_wrapper_mem.py) calls [optimizer_mem.py](optimizer_mem.py).
   - [opt_wrapper_battery.py](opt_wrapper_battery.py) calls [optimizer_battery.py](optimizer_battery.py).
The following guidelines illustrates how to run the optimizer.
## Inputs
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
## Default optimization 
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
### Optimize under memory constrain
#### 4. memory_constrain
Config the device memory constrain (Unit: MB) in [opt_wrapper_mem.py](opt_wrapper_mem.py)
```python
# Change memory config here
memory_constrain = 1024*2 # MB
```
And use [opt_wrapper_mem.py](opt_wrapper_mem.py) instead of the default wrapper.
### Optimize for battery life
#### 5. battery life
Use [opt_wrapper_battery.py](opt_wrapper_battery.py) instead of the default wrapper.
## Outputs
While an optimizer outputs (1) a part.csv that specifies layers and their assigned devices, and (2) a priority.csv that specifies the execution order, a wrapper has different outputs. Since a wrapper finds several partitioning strategies across a range of network bandwidth, we have the results recorded in the [data](data) directory. While using wrappers, the part.csv and priority.csv are merely intermediate files. 
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
3. run `python3 opt_wrapper.py`. Check [data/yolov4-agx.csv](data%2Fyolov4-agx.csv) for performance results across the range of bandwidth. 
4. If you want to run the experiment under a particular bandwidth, please set the bandwidth as such:
```python
# Bandwidths that the optimizer will run through. Categorized by device model. Unit: mbps
bandwidths = {
  'agx':
      {
       'yolov4': [2000],
      }
}
```
Then please check part.csv and priority.csv in [testcases/yolov4-agx](testcases%2Fyolov4-agx) for partitioning results. You may also use the dep.csv, part.csv and priority.csv files in a NS-DOT-Visualizer to visualize the results.  
5. run `python3 power-infer.py` to calculate the energy consumption of data transfer. 
6. run `python3 PLT_energy/PYPLT_energy.py` to visualize the battery life optimization. Check [PLT_energy/<power mode>/yolov4-agx.png](PLT_energy%2F1%2Fyolov4-agx.png) for battery life optimization plots.