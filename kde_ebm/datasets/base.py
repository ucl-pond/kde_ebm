import csv
import numpy as np
from os.path import join
from os.path import dirname


def load_synthetic(data_file_name):
    """ This is almost completely stolen from sklearn!
        Loads data from data/data_file_name.

    Parameters
    ----------
    data_file_name : String. Name of csv file to be loaded from
    module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy Array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy Array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy Array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.

    feature_names : Numpy Array
        A 1D array containing the names of the features. These are used
        in plotting functions later.
    """
    module_path = dirname(__file__)
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        feature_names = ['BM%i' % (x+1) for x in range(n_features)]
        feature_names = np.array(feature_names)
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=int)
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=int)
    return data, target, feature_names, target_names
