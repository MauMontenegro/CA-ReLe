import statistics
import numpy as np
import math as mt
import os


class ExperimentLog:
    def __init__(self, path, file_name):
        self.path = path
        self.file = file_name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.full_path = os.path.join(self.path, self.file)

    def log_save(self, mean_rewards, mean_samples):
        data_log = np.c_[(mean_rewards, mean_samples)]

        np.savetxt(self.full_path, data_log, fmt=["%f", "%f"],
                   delimiter=",")
