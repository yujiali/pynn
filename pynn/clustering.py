"""
Some clustering algorithms.

Yujia Li, 05/2015
"""

import gnumpy as gnp
import numpy as np
import time

########################## initialization #############################

def _init_plus(X, K, dist='euclidean'):
    f_dist = choose_distance_metric(dist)
    C = X[np.random.randint(X.shape[0])].reshape(1,-1)

    for k in xrange(1, K):
        idx = f_dist(X, C).min(axis=1).argmax()
        C = gnp.concatenate([C, X[idx].reshape(1,-1)], axis=0)

    return C

def _init_sample(X, K, dist='euclidean'):
    idx = np.random.permutation(X.shape[0])
    return X[idx[:K]]

def _init_random(X, K, dist='euclidean'):
    x_max = X.max(axis=0).reshape(1,-1)
    x_min = X.min(axis=0).reshape(1,-1)

    return gnp.rand(K, X.shape[1]) * (x_max - x_min) + x_min

def choose_initializer(method):
    if method == 'plus':
        return _init_plus
    elif method == 'sample':
        return _init_sample
    elif method == 'random':
        return _init_random
    else:
        raise Exception('Initialization method "%s" is not supported.' % method)

########################## distance metric ############################

def _dist_euclidean(X, Y):
    """
    d_ij = (x_i - y_j)^2
    """
    X = gnp.as_garray(X)
    Y = gnp.as_garray(Y)
    X_diag = (X*X).sum(axis=1)
    Y_diag = (Y*Y).sum(axis=1)

    return gnp.sqrt(-2 * X.dot(Y.T) + X_diag.reshape(-1,1) + Y_diag.reshape(1,-1) + 1e-3)

def _dist_cosine(X, Y):
    """
    d_ij = 0.5 * (1 - <x_i,y_j> / |x_i| |y_j|)

    This is a variant of the cosine similarity, modified so that it is a
    distance, i.e. the smaller the closer, and the distance is between 0 and 1.
    """
    X_norm = gnp.sqrt((X*X).sum(axis=1)) + 1e-10
    Y_norm = gnp.sqrt((Y*Y).sum(axis=1)) + 1e-10

    return (1 - X.dot(Y.T) / (X_norm.reshape(-1,1).dot(Y_norm.reshape(1,-1)))) / 2

def choose_distance_metric(metric_name):
    if metric_name == 'euclidean':
        return _dist_euclidean
    elif metric_name == 'cosine':
        return _dist_cosine
    else:
        raise Exception('Distance metric "%s" is not supported.' % metric_name)

########################## empty action ############################

def kmeans(X, K, init='plus', dist='euclidean', empty_action='singleton', max_iters=100, verbose=True):
    """
    X: NxD dataset, each row is one data point.
    init: method to choose initial cluster centers.  Available options: {
        'plus': k-means++, 
        'sample': randomly sample K data points,
        'random': generate K points uniformly at random from X's range }
    dist: distance metric to be used. Available options: {
        'euclidean': Euclidean distance. }
    empty_action: action to take when one cluster lost all its members.  
        Available options: {
        'singleton': create a new cluster to replace it using a point furthest 
            to the current center.
        'error': raise an exception. }
    max_iters: maximum number of iterations to run.
    verbose: if False, nothing will be printed during training.

    Return:
        C: KxD matrix, cluster centers, each row is one center
        idx: N-d vector, cluster assignments for each data point.
        loss: sum of distances for the dataset under the given distance metric.
    """
    t_start = time.time()
    gnp.free_reuse_cache()
    gnp.max_memory_usage = 3.8 * 1000 * 1000 * 1000

    def f_print(s, newline=True):
        if verbose:
            if newline:
                print s
            else:
                print s,
        else:
            pass

    f_print('Initializing k-means...', newline=False)
    X = gnp.as_garray(X)
    X_cpu = X.asarray().astype(np.float64)
    
    if isinstance(init, str):
        f_init = choose_initializer(init)
        C = f_init(X, K, dist=dist)
    elif isinstance(init, gnp.garray) or isinstance(init, np.ndarray):
        C = gnp.as_garray(init)
        print '[Warning] Init centers provided, K and init not used.'
        K = C.shape[0]

    f_dist = choose_distance_metric(dist)

    loss = 0
    idx = None
    prev_idx = None

    full_idx = np.arange(X.shape[0])
    f_print('done [%.2fs]' % (time.time() - t_start))

    t_start = time.time()
    i_iter = 0
    while i_iter <= max_iters:
        gnp.free_reuse_cache()
        f_print('iter %d,' % i_iter, newline=False)
        
        # use GPU to compute distance because it is fast,
        # bug go back to CPU to avoid low precision problem
        D = f_dist(X, C).asarray().astype(np.float64)
        idx = D.argmin(axis=1)
        loss = D[full_idx, idx].sum()

        if prev_idx is not None and (idx == prev_idx).all():
            print '** k-means converged **'
            break
        else:
            prev_idx = idx

        # update cluster center
        do_restart = False
        for k in xrange(K):
            k_idx = full_idx[idx == k]
            if k_idx.size == 0:
                if empty_action == 'singleton':
                    # update C
                    C[k] = X[f_dist(X, C[k:k+1]).ravel().argmax()]
                    do_restart = True
                elif empty_action == 'error':
                    raise Exception('Empty cluster encountered in k-means!')
                else:
                    raise Exception('Action not specified for empty cluster.')
            else:
                C[k] = X_cpu[k_idx].mean(axis=0)

        f_print('loss=%.2f, [%.2fs]' % (loss, time.time() - t_start))

        if do_restart:
            print '[Warning] restarting because empty clusters encountered.'
            i_iter = 0

        t_start = time.time()

        i_iter += 1

    return C, idx, loss

