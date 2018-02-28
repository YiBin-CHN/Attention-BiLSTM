import initializations
from utils.theano_utils import shared_zeros, alloc_zeros_matrix, sharedX
import theano.tensor as T
import theano
import numpy as np
from utils.numpy_utils import ortho_weight
import theano.sandbox.cuda.basic_ops

class LSTM(object):
    
    """
    input: n_step * n_sample * em_dim
    mask: n_step * n_sample
    output: n_step * n_sample * h_dim
    """
    
    def __init__(self,in_dim, h_dim, name=None, init='orthogonal', soft_init='normal'):
        if name is not None:
            self.set_name(name)
        self.name = name
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.init = initializations.get(init)
        self.soft_init = initializations.get(soft_init)
        self.init_params()

    def set_name(self,name):
        if not isinstance(name,basestring):
            raise TypeError('name must be string.')
        self.name = name
        
    def pack_params(self):
        self.params = [self.W,self.U,self.b]
        self.regulariable = [self.W,self.U]
        
    def load_params(self,model):
        self.W = sharedX(model[0])
        self.U = sharedX(model[1])
        self.b = sharedX(model[2])
        self.pack_params()

    def init_params(self):
                
        W = self.init((self.in_dim, self.h_dim))
        self.W = np.concatenate([ortho_weight((self.in_dim, self.h_dim)),
                                 ortho_weight((self.in_dim, self.h_dim)),
                                 ortho_weight((self.in_dim, self.h_dim)),
                                 ortho_weight((self.in_dim, self.h_dim))
                                 ],
                                axis = 1
                                )
        self.W = sharedX(self.W)
        self.U = np.concatenate([
                                 ortho_weight((self.h_dim, self.h_dim)),
                                 ortho_weight((self.h_dim, self.h_dim)),
                                 ortho_weight((self.h_dim, self.h_dim)),
                                 ortho_weight((self.h_dim, self.h_dim))
                                 ],
                                axis = 1
                                )
        self.U = sharedX(self.U)
        
        self.b = shared_zeros((4*self.h_dim,))
        
        self.pack_params()
        
    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    
    def _step(self, m_t, x_t, h_tm1, c_tm1):
        W = self.W 
        U = self.U 
        b = self.b
        
        preact = T.dot(h_tm1, U)
        state_below = T.dot(x_t,W) + b 
        preact += state_below
        
        i = T.nnet.sigmoid(self._slice(preact, 0, self.h_dim))
        f = T.nnet.sigmoid(self._slice(preact, 1, self.h_dim))
        o = T.nnet.sigmoid(self._slice(preact, 2, self.h_dim))
        c = T.tanh(self._slice(preact, 3, self.h_dim))
        
        c = f * c_tm1 + i * c
        c = m_t[:, None] * c + (1. - m_t)[:, None] * c_tm1

        h = o * T.tanh(c)
        h = m_t[:, None] * h + (1. - m_t)[:, None] * h_tm1
        
        return h, c
        
                
    def get_outputs(self,X,mask=None,h0=None,c0=None): # need h0,c0, how to make it...
        """
        X: Nt * batch_size * dim
        mask: Nt * batch_size   
        """
        assert mask is not None
        assert X.ndim==3 # need to modified to adaptive for 2-d input.
        
        n_samples = X.shape[1]

        if h0 is None:
            h_tm1 = T.unbroadcast(alloc_zeros_matrix(n_samples, self.h_dim), 1)
        else:
            h_tm1 = T.unbroadcast(h0,1)
        if c0 is None:
            c_tm1 = T.unbroadcast(alloc_zeros_matrix(n_samples, self.h_dim), 1)
        else:
            c_tm1 = T.unbroadcast(c0,1)
        
        [h, c], updates = theano.scan(self._step,
                                      sequences=[mask, X],
                                      outputs_info=[h_tm1,c_tm1]
                                       )
        
        return h,c
        
    

    
        
