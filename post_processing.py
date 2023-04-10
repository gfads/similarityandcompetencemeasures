class PickleStructure:
    def __init__(self,
                 deployment,
                 horizontal_step,
                 lags,
                 level_grid: str,
                 metric,
                 model_type,
                 training,
                 trained_model,
                 testing,
                 time_series,
                 time_window_size,
                 scaler,
                 validation,
                 workload,
                 ):

        self.deployment = deployment,
        self.folder = workload + '/' + metric + '/' + deployment + '/' + level_grid + '/'
        self.model_name = model_type + time_window_size
        self.file_path = self.folder + self.model_name
        self.horizontal_step = horizontal_step
        self.lags = lags
        self.level_grid = level_grid
        self.metric = metric
        self.model_type = model_type
        self.training = training
        self.trained_model = trained_model
        self.testing = testing
        self.time_series = time_series
        self.time_window_size = time_window_size
        self.scaler = scaler
        self.validation = validation
        self.workload = workload

        if level_grid == 'bagging':
            self.folder = workload + '/' + metric + '/' + deployment + '/homogeneous/' + level_grid + '/'
            self.save_bagging()
        else:
            save_pickle(self.__dict__)

    def save_bagging(self):
        from copy import deepcopy
        models = deepcopy(self.trained_model)

        for model_type, value in models.items():
            self.trained_model = value['model']
            self.model_name = model_type + self.time_window_size
            self.file_path = self.folder + self.model_name

            save_pickle(self.__dict__)


def save_pickle(df: dict):
    from pickle import dump
    from os import makedirs
    from os.path import dirname

    makedirs(dirname('pickle/' + df['folder']), exist_ok=True)
    dump(df, open('pickle/' + df['file_path'] + '.pkl', 'wb'))


def load_pickle(file_path: str):
    from pickle import load

    return load(open('pickle/' + file_path, 'rb'))


def save_model(deployment: str,
               lags: int,
               level_grid: str,
               metric: str,
               model_type: str,
               training: list,
               trained_model: object,
               testing: list,
               time_series: list,
               time_window_size: str,
               scaler: object,
               workload: str,
               horizontal_step: int = 1,
               validation: list = None):
    """
    :param validation:
    :param testing:
    :param training:
    :param workload:
    :param metric:
    :param deployment:
    :param lags:
    :param scaler:
    :param time_series:
    :param time_window_size:
    :param trained_model:
    :param model_type:
    :param horizontal_step:
    :param level_grid:
    """

    PickleStructure(deployment, horizontal_step, lags, level_grid, metric, model_type, training, trained_model, testing,
                    time_series, time_window_size, scaler, validation, workload)
