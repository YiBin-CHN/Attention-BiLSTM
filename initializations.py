from utils.theano_utils import sharedX, shared_zeros, shared_ones
import numpy as np

def uniform(shape, scale=0.05):
    return sharedX(np.random.uniform(low=-scale, high=scale, size=shape))

def normal(shape, scale=0.05):
    return sharedX(np.random.randn(*shape)*scale)

def xavier(shape,scale=None):
  var_w = 2./(shape[0]+shape[1])
  return sharedX(np.random.normal(0.,np.sqrt(var_w),size=shape))
  
def orthogonal(shape, scale=1.1):
    ''' From Lasagne. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return sharedX(scale * q[:shape[0], :shape[1]])

def zero(shape):
    return shared_zeros(shape)


def one(shape):
    return shared_ones(shape)

from utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'initialization')