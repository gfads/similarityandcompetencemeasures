from itertools import product
from pickle_functions import load_pickle
from pandas import DataFrame

series = ['alibaba']

# Generate Competence Measures Results
competence_measures = ['variance', 'sum_absolute_error', 'sum_squared_error', 'min_squared_error', 'max_squared_error',
                       'neighbors_similarity', 'rmse', 'closest_squared_error', 'svr', 'oracle']
distance_measures = ['euclidean']

# Generate Similarity Measures Results
#competence_measures = ['rmse']
#distance_measures = ['cityblock', 'chebyshev', 'cosine', 'correlation', 'dtw', 'edrs', 'euclidean', 'hellinger', 'gower', 'shape_dtw', 'svr', 'oracle']

metrics = ['traffic']
microservices = [str(i) for i in range(1, 9)]
window_size = ['20']

parameters = list(product(competence_measures, metrics, microservices, series, window_size))
data = []
for dm in distance_measures:
    for cm in competence_measures:
        line = []
        for me in metrics:
            for mi in microservices:

                if dm == 'svr' or cm == 'svr':
                    mse = load_pickle(series[0] + mi + me + 'svr20')['mse']

                    line.append(mse)

                elif dm == 'oracle' or cm == 'oracle':
                    line.append(load_pickle(series[0] + mi + me + '20oracle')['mse'])
                else:
                    path = series[0] + mi + me + window_size[0] + cm + dm + '10ds'
                    model = load_pickle(path)

                    line.append(model['mse'])

        data.append(line)

DataFrame(data).to_csv('results.csv', index=competence_measures)
