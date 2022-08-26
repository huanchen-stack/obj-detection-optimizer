import numpy as np
import pandas as pd
import sys
import os
import pickle
from opt_wrapper import OPT_WRAPPER
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, SCORERS
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor


class power_inferer(object):
    def __init__(self):
        self.config = OPT_WRAPPER.configs
        self.result = []

        for config in self.config:
            self.result.clear()
            df = pd.read_csv(self.get_path(config))
            df.columns = ["bandwidth", "optimizer", "energy", "device", "payload"]
            for i in range(len(df)-1):
                self.tempfile = open("input.csv", "w")
                self.tempfile.write("nr_ssRsrp,avg_power,downlink_mbps,uplink_mbps,data_size\n")
                self.tempfile.write(f"0.75,0,{df['bandwidth'][i]},0,{df['payload'][i]}\n")
                self.tempfile.close()
                res = self.predict("down")

                self.tempfile = open("input.csv", "w")
                self.tempfile.write("nr_ssRsrp,avg_power,downlink_mbps,uplink_mbps,data_size\n")
                self.tempfile.write(f"0.75,0,0,{df['bandwidth'][i]},{df['payload'][i]}\n")
                self.tempfile.close()
                res += self.predict("up")

                df.at[i, 'energy'] = res
            df.to_csv(self.get_path(config), sep=',', index=False)

    def calculate_energy(self, data_size, power, speed):
        if speed == 0:
            return 0
        return data_size / speed * power

    def get_path(self, config):
        path = os.path.abspath(os.getcwd())
        return os.path.join(path, f"PLT_energy/{config}.csv")

    def predict(self, type):

        df = pd.read_csv("input.csv")

        df.columns = ["nr_ssRsrp", "avg_power", "downlink_mbps", "uplink_mbps", "data_size"]
        X_column = ["downlink_mbps", "uplink_mbps"]

        data_size = df["data_size"].to_numpy()[0] * 8
        speed_up = df["uplink_mbps"].to_numpy()[0]
        speed_down = df["downlink_mbps"].to_numpy()[0]

        X = df[X_column].to_numpy()

        dtr_vz_mn_model = pickle.load(open("power_infer/dtr_vz_mn_model.pickle", "rb"))

        e_mn = dtr_vz_mn_model.predict(X)
        if type == "down":
            res = self.calculate_energy(data_size, e_mn[0], speed_down)
        else:
            res = self.calculate_energy(data_size, e_mn[0], speed_up)
        return res


if __name__ == '__main__':
    power_inferer()