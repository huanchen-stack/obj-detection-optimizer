import pandas as pd
import os

MEM_CONSTRAIN = 1024 * 2
# MEM_CONSTRAIN = 1024 * 4

READ_SPEED = 1280
# READ_SPEED = 190

def calculate(config):
    res = {}
    for b in range(250, 2500, 250):
        print(b)
        device_dict, mem_dict = load_partition_and_mem(config, b)
        for i in range(0,2):
            sum = 0
            for layer in device_dict[i]:
                sum += mem_dict[layer]
            if sum > MEM_CONSTRAIN:
                res[b] = (sum - MEM_CONSTRAIN)/READ_SPEED
            else:
                res[b] = 0
    res_file = open(os.path.join("part.csv"), "w")
    res_file.write(f"bandwidth,device\n")
    for layer_for_csv in self.layers:
        self.partitions.write(f"{layer_for_csv},{self.layers[layer_for_csv].device_id}\n")
    self.partitions.close()

def load_partition_and_mem(config, bandwidth):
    device_dict = {
        0: [],
        1: [],
    }
    mem_dict = {}
    part_list = pd.read_csv(f"{config}/part-{bandwidth}.csv").values.tolist()
    for entry in part_list:
        layer = entry[0]
        part = entry[1]
        device_dict[part].append(layer)
    prof_list = pd.read_csv(f"../../testcases/{config}/1/prof.csv").values.tolist()
    for entry in prof_list:
        layer = entry[0]
        mem = entry[3]
        mem_dict[layer] = mem
    return device_dict, mem_dict

if __name__ == '__main__':
    configs = [
        'faster-agx',
        # 'faster-nano',
        # 'yolor-agx',
        # 'yolor-nano',
        # 'yolox-agx',
        # 'yolox-nano',
        # 'yolov4-agx',
        # 'yolov4-nano',
        # 'yolos-agx'
    ]
    for config in configs:
        calculate(config)
