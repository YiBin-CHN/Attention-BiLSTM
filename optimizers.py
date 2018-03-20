import theano.tensor as T
from utils.theano_utils import shared_scalar, shared_zeros
from utils.generic_utils import get_from_module
from theano import function, config, shared, sandbox
import theano.sandbox.cuda.basic_ops
import numpy as np

def clip_l2(grad,threshold):
    g_l2 = np.sqrt((grad**2).sum())
    return T.switch(T.lt(threshold,g_l2),grad*threshold/g_l2,grad)
    """
    if g_l2>threshold:
        return grad*threshold/g_l2
    else:
        return grad
    """
    
class Optimizers(object):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.updates = []
        
    def get_updates(self, loss, params):
        raise NotImplementedError
    
    def get_gradients(self, loss, params):
        
        grads = T.grad(loss, params)
        
        return grads    
    
class Adadelta(Optimizers):
    """
        adadelta algorithm
    """
    def __init__(self, lr=1.0, rho=0.95, epsilon=1e-6, *args, **kwargs):
        super(Adadelta, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.lr = shared_scalar(lr)
        
    def get_updates(self, loss, params,t=10):
        grads = self.get_gradients(loss, params)
        grad_c = []
        for g in grads:
            grad_c.append(clip_l2(g,t))
        accumulators = [shared_zeros(p.get_value().shape) for p in params]
        delta_accumulators = [shared_zeros(p.get_value().shape) for p in params]
        self.updates = []
        
        for p, g, a, d_a in zip(params, grad_c, accumulators, delta_accumulators):
            new_a = self.rho * a + (1-self.rho) * g**2   #update delta
            self.updates.append((a, new_a))
            
            # use the new accumulator and the *old* delta_accumulator
            delta = (-g) * T.sqrt(d_a + self.epsilon) / T.sqrt(new_a + self.epsilon)
            new_p = p + self.lr * delta
            self.updates.append((p, new_p))
            
            # update delta_accumulator
            new_d_a = self.rho * d_a + (1-self.rho) * delta**2
            self.updates.append((d_a, new_d_a))
            
        return self.updates
    
class SGD(Optimizers):
    def __init__(self, lr=0.01, momentum=0, decay=0., nesterov=False, *args, **kwargs):
        super(SGD, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = shared_scalar(0)
        self.lr = shared_scalar(lr)
        self.momentum = shared_scalar(momentum)
        self.decay = shared_scalar(decay)
        
    def get_updates(self, loss, params,t=10):
        
        grads = self.get_gradients(loss, params)
        grads_clipped = []
        for g in grads:
            grads_clipped.append(clip_l2(g,t))
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        self.updates = [(self.iterations, self.iterations + 1.)]
        for p, g in zip(params, grads_clipped):
            m = shared_zeros(p.get_value().shape)  # momentum
            v = self.momentum * m - lr * g  # velocity
            self.updates.append((m, v))
            
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            self.updates.append((p, new_p))
        
        return self.updates

    
# aliases
adadelta = Adadelta
sgd = SGD

def get(identifier, kwargs=None):
    return get_from_module(identifier, globals(), 'optimizer', instantiate=True, kwargs=kwargs)
            
