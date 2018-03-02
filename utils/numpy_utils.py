import theano
import numpy as np
import theano.tensor as T
import theano

def numpyX(X, dtype=theano.config.floatX):
    return np.asarray(X, dtype=dtype)

def ortho_weight(shape, scale=1.1):
    
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    
    return scale * q[:shape[0], :shape[1]]