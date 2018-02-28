import initializations
from utils.theano_utils import shared_zeros,sharedX
import activations
import theano.tensor as T
import theano


class Dense(object):
    
    """
    input: n_samples * in_dim// Nt * batch_size * dim
    output: n_samples * out_dim// Nt * batch_size * dim
    """
    
    def __init__(self, in_dim,  out_dim, name=None,init='xavier', activation='relu'):

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
        self.params = [self.W,self.b]
        self.regulariable = [self.W]
        
    def load_params(self,model):
        self.W = sharedX(model[0])
        self.b = sharedX(model[1])
        self.pack_params()

    def init_params(self):
        self.W = self.init((self.in_dim, self.out_dim))
        self.b = shared_zeros((self.out_dim,))
        self.pack_params()
        
    def _step(self,x_t):
        return self.activation(T.dot(x_t,self.W) + self.b)
        
    def get_outputs(self,X):
        """
        X: Nt * n_samples * in_dim
        """
        outputs,updates = theano.scan(self._step,sequences=[X])
        
        return outputs
        
class scale_fac(object):
    def __init__(self, in_dim,  name=None,init='uniform', activation='linear'):

        if name is not None:
            self.set_name(name)
        self.name = name
        self.in_dim = in_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)    
        self.init_params()
        
    def init_params(self):
      self.W = self.init((self.in_dim,),scale=1.0)
      self.pack_params()
    
    def pack_params(self):
        self.params = [self.W]

    def load_params(self,model):
        self.W = sharedX(model[0])
        self.pack_params()
        
    def _step(self,x_t):
        return self.W*x_t
        
    def get_outputs(self,X):
        outputs,updates = theano.scan(self._step,sequences=[X])
        
        return outputs        