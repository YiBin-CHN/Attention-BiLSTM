import initializations
from utils.theano_utils import shared_zeros, sharedX
import activations
import theano.tensor as T
import theano
import theano.sandbox.cuda.basic_ops

class Embedding(object):
    
    """
    input:
        inputdata: n_samples * in_dim
                or Nt * n_samples * in_dim
        in_dim:
        out_dim:
    output: n_samples * out_dim
        or Nt * n_samples * out_dim
    """
    
    def __init__(self, in_dim=100, out_dim=100, name=None, init='uniform', activation='linear'):
        
        # n_samples * in_dim
        if name is not None:
            self.set_name(name)
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.init_params()
        
    def set_name(self,name):
        if not isinstance(name,basestring):
            raise TypeError('name must be string.')
        self.name = name
        
    def pack_params(self):
        self.params = [self.W]
        self.regulariable = [self.W]
        
    def load_params(self,model):
        self.W = sharedX(model[0])
        self.pack_params()
        
    def init_params(self):
        self.W = self.init((self.in_dim, self.out_dim))
        self.pack_params()
        
    def _step(self,x_t):
        return self.activation(T.dot(x_t,self.W))
        
    def get_outputs(self,X):        
        """
        X: Nt * Nsamp * dim
        """
        outputs,updates = theano.scan(self._step,sequences=[X])
        return outputs