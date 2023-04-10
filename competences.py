def calculate_model_accuracy(y_true, y_pred, measure: str, sample_weight=None):
    from statistics import variance

    if measure == 'mse':
        return mse(y_true, y_pred)
    elif measure == 'variance':
        return variance(y_pred)
    elif measure == 'sum_absolute_error':
        return sum_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'sum_squared_error':
        return sum_squared_error(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'min_squared_error':
        return min_squared_error(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'max_squared_error':
        return max_squared_error(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'neighbors_similarity':
        return neighbors_similarity(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'rmse':
        return rmse(y_true, y_pred, sample_weight=sample_weight)
    elif measure == 'closest_squared_error':
        return closest_squared_error(y_true, y_pred)
    else:
        return 'The metric is yet not implemented'


def rmse(y_true, y_pred, sample_weight=None):
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False)


def sum_absolute_error(y_true, y_pred, sample_weight):
    from numpy import sum, abs

    error = 0
    for i in range(0, len(y_true)):
        error = error + (abs(y_true[i] - y_pred[i]) * sample_weight[i])

    return error / sum(sample_weight)


def sum_squared_error(y_true: list, y_pred: list, sample_weight):
    from numpy import sum

    error = 0

    for i in range(0, len(y_true)):
        error = error + (((y_true[i] - y_pred[i]) ** 2) * sample_weight[i])

        return error / sum(sample_weight)


def min_squared_error(y_true: list, y_pred: list, sample_weight):
    from numpy import min

    error = []
    for i in range(0, len(y_true)):
        error.append(((y_true[i] - y_pred[i]) ** 2) * sample_weight[i])

    return min(error)


def max_squared_error(y_true: list, y_pred: list, sample_weight):
    from numpy import max

    error = []
    for i in range(0, len(y_true)):
        error.append(((y_true[i] - y_pred[i]) ** 2) * sample_weight[i])

    return max(error)


def neighbors_similarity(y_true, y_pred, sample_weight):
    error = 0
    for i in range(0, len(y_true)):
        error = error + (((y_true[i] - y_pred[i]) ** 2) * sample_weight[i])

    return error / sum(sample_weight)


def closest_squared_error(y_true, y_pred):
    from numpy import subtract, float64

    return subtract(y_true, y_pred, dtype=float) ** 2


def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true, y_pred)
