import numpy as np


def competence_region_definition(cr_len, training, testing, distance_measure):
    dt_x_cr = {}
    dt_y_cr = {}
    dt_do_cr = {}

    for icr in range(0, len(testing)):
        distance_cr = calculate_the_distance_between_the_windows(training, testing[icr, 0:-1], distance_measure)
        x_cr, y_cr, indices, do_cr = collect_the_competence_region(training, distance_cr, cr_len)

        dt_x_cr[icr] = x_cr
        dt_y_cr[icr] = y_cr
        dt_do_cr[icr] = do_cr

    return dt_x_cr, dt_y_cr, dt_do_cr


def collect_the_competence_region(training_sample, distance_cr, len_competence_region):
    indices_patterns = range(0, len(training_sample))
    distance_cr, indices_patterns = zip(
        *sorted(zip(distance_cr, indices_patterns)))
    indices_patterns_l = list(indices_patterns)

    cr_x = training_sample[:, 0:-1][indices_patterns_l[0:len_competence_region]]
    cr_y = training_sample[:, -1][indices_patterns_l[0:len_competence_region]]

    return cr_x, cr_y, indices_patterns_l[0:len_competence_region], distance_cr[0:len_competence_region]


def calculate_the_distance_between_the_windows(training_sample, testing_sample, measure):
    competence_region = []

    d = None

    for i_training in range(0, len(training_sample)):
        if measure == 'cityblock':
            d = cityblock(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'chebyshev':
            d = chebyshev(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'cosine':
            d = cosine(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'correlation':
            d = correlation(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'dtw':
            d = dtw(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'edrs':
            d = edrs(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'euclidean':
            d = euclidean(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'hellinger':
            d = hellinger(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'gower':
            d = gower(testing_sample, training_sample[i_training, 0:-1])
        elif measure == 'shape_dtw':
            d = shape_dtw(testing_sample, training_sample[i_training, 0:-1])
        else:
            print('métrica não implementada!')

        competence_region.append(d)

    return competence_region


def cosine(a, b):
    from scipy.spatial.distance import cosine

    return cosine(a, b)


def correlation(a, b):
    from scipy.spatial import distance

    return distance.correlation(a, b)


def chebyshev(a, b):
    from scipy.spatial.distance import chebyshev

    return chebyshev(b, a)


def cityblock(a, b):
    from scipy.spatial.distance import cityblock

    return cityblock(a, b)


def dtw(a, b):
    from dtaidistance import dtw

    d = dtw.distance_fast(a, b)

    return d


def edrs(a, b):
    import edit_distance

    a = a.tolist()
    b = b.tolist()

    sm = edit_distance.SequenceMatcher(a, b)
    return sm.distance()


def euclidean(a, b):
    from scipy.spatial.distance import euclidean

    return euclidean(a, b)


def hellinger(p, q):
    diff = 0
    for i in range(0, len(p)):
        diff += (np.sqrt(p[i]) - np.sqrt(q[i])) ** 2
    return 1 / np.sqrt(2) * np.sqrt(diff)


def gower(a, b):
    from gower import gower_matrix

    a = np.array(a, ndmin=2)
    b = np.array(b, ndmin=2)

    return gower_matrix(a, b)[0][0]


def shape_dtw(a, b):
    a = np.reshape(a, (len(a), 1))
    b = np.reshape(b, (len(b), 1))

    dist, correspondences = matching_shapedtw(a, b, euclidean)
    return dist


def matching_shapedtw(x, y, dist):
    from fastdtw import fastdtw
    from shape_context import find_ranges, compute_shape_context

    """ matches a two time series of dim-dimensional data x and y by FastDTW.
        Input:
            x: A time series of a dim-dimensional data as a 2d numpy.array,
                where x[i, :] denotes its i-th data.
            y: A time series of a dim-dimensional data as a 2d numpy.array,
                where y[i, :] denotes its i-th data.
        Return:
            dist: The total distance between the sequentes x and y.
            correspondences: A set of established correspondences,
            where x[correspondences[i][0], :] and y[correspondences[i][1]] are
            i-th corresponding points.
    """
    # assert x.ndim == y.ndim
    # assert len(x[0, :]) == len(y[0, :])

    # compute shape context
    ranges = find_ranges(np.concatenate((x, y)))
    feature_x = compute_shape_context(x, ranges)
    feature_y = compute_shape_context(y, ranges)

    # reshape the feature tensor of arbitrary order to matrix
    # because fastdtw only accepts 2d numpy.arrays.
    return fastdtw(feature_x.reshape(len(feature_x), -1), feature_y.reshape(len(feature_y), -1), dist=dist)
