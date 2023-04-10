def find_better_model(training_models):
    from numpy import Inf

    best_result, best_model = Inf, Inf

    for tm in training_models:
        actual_model = tm[0]
        actual_result = tm[1]

        if actual_result < best_result:
            best_result = actual_result
            best_model = actual_model

    return best_model


def pipeline_training_model(models, training, competence_measure, validation=None):
    from c1copy import calculate_model_accuracy
    x_train, y_train = training[:, 0:-1], training[:, -1]

    for model in models:
        model.fit(x_train, y_train)

    if validation is not None:
        cm_models = []
        for model in models:
            x_val, y_val = validation[:, 0:-1], validation[:, -1]
            predicted = model.predict(x_val)
            accuracy_metric = calculate_model_accuracy(y_val, predicted, competence_measure)
            cm_models.append([model, accuracy_metric])

        model = find_better_model(cm_models)

        return model

    else:
        return models[0]


def svr_train(training, competence_measure='mse', grid_level: str = 'default', bagging_size=0, validation=None):
    from sklearn.svm import SVR

    if grid_level == 'default':
        model = SVR()
        model = pipeline_training_model([model], training, competence_measure=competence_measure)

        return model
    elif grid_level == 'hard':
        from itertools import product

        kernel = ['rbf', 'sigmoid']
        gamma: list = [0.5, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        regularization_parameter: list = [0.1, 1, 100, 1000, 10000]
        epsilon: list = [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001]

        hyper_param = list(product(kernel, gamma, epsilon, regularization_parameter))

        models = []
        for k, g, e, rp in hyper_param:
            models.append(SVR(C=rp, epsilon=e, kernel=k, gamma=g))

        return pipeline_training_model(models, training, competence_measure=competence_measure, validation=validation)

    elif grid_level == 'bagging':
        models = bagging(bagging_size, training, validation, 'svr')

        return models


def resampling(serie, n):
    import numpy as np
    size = len(serie)
    ind_particao = []

    for i in range(n):
        ind_r = np.random.randint(size)
        ind_particao.append(ind_r)

    return ind_particao


def bagging(qtd_modelos, training, validation, name_model):
    models = {'model': [], 'training_sample': [], 'validation_sample': [], 'indices': []}

    for i in range(qtd_modelos):
        models[name_model + str(i)] = {}
        indices = resampling(training, len(training))
        particao = training[indices, :]

        if name_model == 'svr':
            models['model'].append(svr_train(particao, grid_level='hard', validation=validation))

        models['training_sample'].append(particao)
        models['validation_sample'].append(validation)
        models['indices'].append(indices)

    return models
