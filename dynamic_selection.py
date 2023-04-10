import sys
from post_processing import load_pickle
from numpy import concatenate, Inf
from competences import calculate_model_accuracy
from pickle_functions import save_pickle_result
from itertools import product
from similarities import competence_region_definition


def load_model(microservice, metric):
    models = {}

    for index in range(0, 100):
        model = load_pickle('alibaba' + microservice + metric + 'svrbagging' + str(index) + '20.pkl')
        models['svr' + str(index)] = {'model': model['model'], 'lags': model['lags']}

    return models


BAGGING_SIZE: int = 100
COMPETENCE_MEASURE: str = 'mse'
METRICS: list = ['traffic']
#MICROSERVICES: list = ['1', '2', '3', '4', '5', '6', '7', '8']
MICROSERVICES: list = ['1']
#MICROSERVICES: list = ['1', '2']
#MICROSERVICES: list = ['3', '4']
#MICROSERVICES: list = ['5', '6']
#MICROSERVICES: list = ['7', '8']
MODEL_NAMES: list = ['svr']
SERIES: list = ['alibaba']
WINDOW_SIZE: list = [20]
RoC = 10

#distance_measure = ['cityblock', 'chebyshev', 'cosine', 'correlation', 'dtw', 'edrs', 'hellinger', 'gower', 'shape_dtw']
distance_measure = ['euclidean']

competence_measure = ['variance']#, 'sum_absolute_error', 'sum_squared_error', 'min_squared_error', 'max_squared_error',
                      #'neighbors_similarity', 'rmse', 'closest_squared_error']

#competence_measure = ['rmse']


parameters = list(product(competence_measure, distance_measure, METRICS, MICROSERVICES, SERIES, WINDOW_SIZE))

for cm, dm, me, mi, s, ws in parameters:

    path = s + mi + me + MODEL_NAMES[-1] + str(WINDOW_SIZE[-1])

    monolithic = load_pickle(path + '.pkl')
    dataset = concatenate((monolithic['training_sample'], monolithic['validation_sample']))
    testing = monolithic['testing_sample']

    cr, crtv, distance = competence_region_definition(RoC, dataset, testing, dm)

    models = load_model(mi, me)
    names, prediction, target, better_models = [], [], [], []

    for i in cr:
        print(f'{s}{mi}{me}{WINDOW_SIZE[-1]}{cm}{dm}{RoC}ds: {((i / len(testing)) * 100):.2f}%')
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')

        best = Inf
        model_name = ''
        better_model = None
        lags = None

        for m in models:
            y_pred = models[m]['model'].predict(cr[i][:, models[m]['lags'][:-1]])
            y_true = crtv[i]

            if cm == 'neighbors_similarity':
                y_pred = [models[m]['model'].predict(testing[i, models[m]['lags'][:-1]].reshape(1, -1))[0]] * 10

            elif cm == 'closest_squared_error':
                y_pred, y_true = [y_pred[0]], [y_true[0]]

            model_accuracy = calculate_model_accuracy(y_true, y_pred, measure=cm, sample_weight=distance[i])

            if model_accuracy <= best:
                best = model_accuracy
                model_name = m
                better_model = models[m]['model']
                lags = models[m]['lags']

        prediction.append(better_model.predict(testing[i, lags[:-1]].reshape(1, -1)))
        target.append(testing[i, -1])
        names.append(model_name)
        better_models.append(better_model)

    save_pickle_result(prediction, target, names, better_models, s + mi + me + str(WINDOW_SIZE[-1]) + cm + dm + str(RoC)
                       + 'ds')
    print(f'{s}{mi}{me}{WINDOW_SIZE[-1]}{cm}{dm}{RoC}ds: 100%')
