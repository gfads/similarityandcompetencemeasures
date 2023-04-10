from pickle_functions import save_model
from models import svr_train
from preprocess import transform_data
from itertools import product


def train_models(
        bagging_size: int = 100,
        competence_measure: str = "mse",
        grid_level: str = 'default',
        metrics: list = ['traffic'],
        microservices: list = ['1'],
        model_names: list = ['svr'],
        series: list = ['alibaba'],
        window_sizes: list = [20]
):
    parameters = list(product(metrics, microservices, series, window_sizes))

    for me, mi, s, ws in parameters:

        path = 'time_series/' + s + '/microservice ' + mi + '/' + me + '.csv'
        training, validation, testing, total, lags, scaler = transform_data(path, ws, 0.6, 0.2)

        for model_name in model_names:
            print(me, mi, s, ws, model_name)

            model = svr_train(training[:, lags], competence_measure=competence_measure, bagging_size=bagging_size + 1,
                              grid_level=grid_level, validation=validation[:, lags])

            save_model(model, model_name, ws, 1, training, testing, total, lags, scaler, grid_level,
                       s + mi + me + model_name + str(ws), validation=validation)


metrics = ['traffic']
microservices = ['1']

print(metrics, microservices)
print('Monolithic Models')
train_models(grid_level='hard', metrics=metrics, microservices=microservices, model_names=['svr'], window_sizes=[20])

print('Bagging Models')
train_models(bagging_size=100, grid_level='bagging', metrics=metrics, microservices=microservices, model_names=['svr'],
             window_sizes=[20])
