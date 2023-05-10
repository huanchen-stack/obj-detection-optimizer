"""
 This Python script estimates the energy for drone communication with a pretrained ML model.
 This scipt needs to be executed after the optimization process,
     since the scipt needs to know which of the intermediate tensors
     need to be transfered.
 This scipt automatically reads the data in the PLT_energy folder.
 We take care of multi-casting and make it an option.
     The default is to use multi-casting.
     There are barely any differences if # drones in use are small.
     
 We assume no compression methods are adopted.
 We also assume that we can always use the full bandwidth.
 """

import os
import pickle

import numpy as np
import pandas as pd

from opt_wrapper import OPT_WRAPPER


class EnergyInferer(object):

    def __init__(self, config, multicast=True):
        self.config = config
        self.dtr_vz_mn_model = pickle.load(open("power_infer/dtr_vz_mn_model.pickle", "rb"))
        self.multicast = multicast
        self.latency = {}

    def get_path(self, config):
        path = os.path.abspath(os.getcwd())
        return os.path.join(path, f"data/{config}.csv")

    def predict_POW(self, uplink_mbps, downlink_mbps):
        X = np.array([uplink_mbps, downlink_mbps]).reshape(1, 2)
        POW_mn = self.dtr_vz_mn_model.predict(X)
        
        # uplink and downlink energy are estimated separately
        assert uplink_mbps == 0 or downlink_mbps == 0, "Either uplink_mbps or downlink_mbps must be zero."
        assert uplink_mbps != 0 or downlink_mbps != 0, "Only one of uplink_mbps or downlink_mbps can be zero."
        
        return POW_mn

    def predict_energy(self):
        """
        This function handles the energy prediction.
        Energy for each transfer is infered separately using ML mode then aggregated.
        We take care of multi-casting as an option.
        """
        
        df = pd.read_csv(self.get_path(self.config))
        df.columns = ["bandwidth", "optimizer", "energy", "device", "payload"]

        for i in range(len(df)):
            # for uplink_mbps and downlink_mbps
            bandwidth = df['bandwidth'][i]
            # get specific datasizes for multicasting
            transfer_data_summary_raw = df['payload'][i].replace('|', ',')
            transfer_data_summary = eval(transfer_data_summary_raw)
            multicaster = {}
            for _, d in transfer_data_summary.items():
                if d['count'] not in multicaster:
                    multicaster[d['count']] = 0
                multicaster[d['count']] += d['size']

            total_energy = 0
            self.latency[self.config] = 0
            for mult, size in multicaster.items():
                bandwidth_pseudo = bandwidth * mult
                POW_up = self.predict_POW(uplink_mbps=bandwidth_pseudo, downlink_mbps=0)
                POW_down = self.predict_POW(uplink_mbps=0, downlink_mbps=bandwidth_pseudo)
                size_pseudo = size * 8 / mult  # convert Byte to bits; apply multicast
                total_energy += size_pseudo / bandwidth_pseudo * (POW_up + POW_down)
                self.latency[self.config] += mult*size/bandwidth
            print(self.latency)
            df.at[i, 'energy'] = total_energy

        df.to_csv(self.get_path(self.config), sep=',', index=False)


def driver(config, multicast):
    inferer = EnergyInferer(config, multicast)
    inferer.predict_energy()


if __name__ == '__main__':

    for config in OPT_WRAPPER.configs:
        driver(config, True)
