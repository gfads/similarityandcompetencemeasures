from itertools import product
from sys import stdout
from numpy import concatenate, Inf
from pickle_functions import load_pickle, save_pickle_result
from dynamic_selection import load_model
from c1copy import calculate_model_accuracy

BAGGING_SIZE: int = 100
COMPETENCE_MEASURE: str = "mse"
METRICS: list = ['traffic']
MICROSERVICES: list = ['1']
MODEL_NAMES: list = ['svr']
SERIES: list = ['alibaba']
WINDOW_SIZE: list = [20]
RoC = 10

parameters = list(product(METRICS, MICROSERVICES, SERIES, WINDOW_SIZE))

for me, mi, s, ws in parameters:
    path = s + mi + me + MODEL_NAMES[-1] + str(WINDOW_SIZE[-1])

    monolithic = load_pickle(path + '.pkl')
    dataset = concatenate((monolithic['training_sample'], monolithic['validation_sample']))
    testing = monolithic['testing_sample']

    models = load_model(mi, me)
    names, prediction, target, better_models = [], [], [], []

    for i in range(0, len(testing)):
        print(f'{s}{mi}{me}{WINDOW_SIZE[-1]}: {((i / len(testing))*100):.2f}%')
        stdout.write('\x1b[1A')
        stdout.write('\x1b[2K')

        best = Inf
        model_name = ''
        better_model = None
        predic = Inf
        for m in models:
            y_pred = models[m]['model'].predict(testing[i, models[m]['lags'][:-1]].reshape(1, -1))
            y_true = [testing[i, -1]]

            model_accuracy = calculate_model_accuracy(y_pred, y_true, measure='mse')

            if model_accuracy <= best:
                best = model_accuracy
                model_name = m
                better_model = models[m]['model']
                lags = models[m]['lags']
                predic = y_pred

        prediction.append(predic)
        target.append(testing[i, -1])
        names.append(model_name)
        better_models.append(better_model)

    save_pickle_result(prediction, target, names, better_models, s + mi + me + str(WINDOW_SIZE[-1]) + 'oracle')
