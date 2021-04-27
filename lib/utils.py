import numpy as np
from random import shuffle

def data_spliter(x: np.ndarray, y: np.ndarray, proportion: float):
    """
            Shuffles and splits the dataset (given by x and y) into a training and a test set, while respecting the given proportion of examples to be kept in the traning set.
        Args:
            x: has to be an numpy.ndarray, a matrix of dimension m * n.
            y: has to be an numpy.ndarray, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the training set.
        Returns:
            (x_train, y_train, x_test, y_test) as a tuple of numpy.ndarray
            None if x or y is an empty numpy.ndarray.
            None if x and y do not share compatible dimensions.
        Raises:
            This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        print("x.shape: ", x.shape, " y.shape: ", y.shape )
        return None
    random_zip = list(zip(x.tolist(), y))
    np.random.shuffle(random_zip)
    np.random.shuffle(random_zip)
    np.random.shuffle(random_zip)
    np.random.shuffle(random_zip)
    new_x = []
    new_y = []
    for e1, e2 in random_zip:
        new_x.append(e1)
        new_y.append(e2)
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    proportion_position = int(x.shape[0] * proportion)
    ret_array = []
    ret_array.append(new_x[:proportion_position])
    ret_array.append(new_y[:proportion_position])
    ret_array.append(new_x[proportion_position:])
    ret_array.append(new_y[proportion_position:])
    return np.array(ret_array, dtype=np.ndarray)

def zscore_normalization(x):
    def mean(x):
        return float(sum(x) / len(x))

    def std(x):
        mean = float(sum(x) / len(x))
        f = lambda x: (x - mean)**2
        tmp_lst = list(map(f, x))
        return float(sum(tmp_lst) / len(x)) ** (0.5)

    mean = mean(x)
    standard_deviation = std(x)
    f = lambda x: (x - mean) / standard_deviation 
    return np.array(list(map(f, x)))

def minmax_normalization(x):
    array_min = min(x)
    array_max = max(x)
    diff_max_min = array_max - array_min
    f = lambda x: (x - array_min) / diff_max_min
    return np.array(list(map(f, x)))