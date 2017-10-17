import csv
import numpy as np
from os.path import join
from os.path import dirname


def load_synthetic(data_file_name):
    module_path = dirname(__file__)
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        feature_names = ['bm%i' % x for x in range(n_features)]
        feature_names = np.array(feature_names)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)
    return data, target, target_names, feature_names
