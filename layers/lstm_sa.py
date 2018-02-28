import initializations
from utils.theano_utils import shared_zeros, alloc_zeros_matrix, sharedX
import theano.tensor as T
import theano
import numpy as np
from utils.numpy_utils import ortho_weight
import theano.sandbox.cuda.basic_ops

class LSTM_SA(object):
    
    """
    input: n_step * n_sample * em_dim
    mask: n_step * n_sample
    output: n_step * n_sample * h_dim
    """
    
    def __init__(self,in_dim, h_dim, ctx_dim,pctx_dim,name=None,selector = True, init='orthogonal', soft_init='normal'):
        if name is not None:
            self.set_name(name)
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.pctx_dim = pctx_dim
        self.ctx_dim = ctx_dim
        self.init = initializations.get(init)
        self.soft_init = initializations.get(soft_init)
        self.selector = selector
        self.init_params()
        
    def set_name(self,name):
        if not isinstance(name,basestring):
            raise TypeError('name must be string.')
        self.name = name
        
    def pack_params(self):
        self.params = [self.W,self.U,self.b,self.Wpc,self.Uph,self.bc,self.Ua,self.Wc]
        self.regulariable = [self.W,self.U,self.Wpc,self.Uph,self.Ua,self.Wc]        
        if self.selector:
            self.params.append(self.Wsel)
            self.params.append(self.bsel)
            self.regulariable.append(self.Wsel)
            
    def load_params(self,model):
        self.W = sharedX(model[0])
        self.U = sharedX(model[1])
        self.b = sharedX(model[2])
        self.Wpc = sharedX(model[3])
        self.Uph = sharedX(model[4])
        self.bc = sharedX(model[5])
        self.Ua = sharedX(model[6])
        self.Wc = sharedX(model[7])
        if self.selector:
            self.Wsel = sharedX(model[8]) 
            self.bsel = sharedX(model[9]) 
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
        # attention params
        # e^i = Ua(tanh(Wctx.dot(context) + Uctx.dot(h_tm1) + bctx))
        self.Wpc = self.init((self.ctx_dim,self.pctx_dim))
        self.Uph = self.init((self.h_dim,self.pctx_dim))
        self.bc = shared_zeros((self.pctx_dim,))
        self.Ua = self.init((self.pctx_dim,1))
        self.ba = shared_zeros((1,))   
        
        self.Wc = self.init((self.ctx_dim,self.h_dim*4))
        if self.selector:
            self.Wsel = self.init((self.h_dim,1))  # if Wsel=h_dim*h_dim  what will happen, is it mean that it could select different feature for different sample?
            self.bsel = shared_zeros((1,))
        self.pack_params()

        
    def _slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]
    
    def _step(self, m_t, x_t, h_tm1, c_tm1,p_ctx,context,mask_c):
        W = self.W 
        U = self.U 
        b = self.b
        
        Uph = self.Uph
        Ua = self.Ua
        ba = self.ba
        Wc = self.Wc
        
        p_h = T.dot(h_tm1,Uph)
        p_a = T.tanh(p_h+p_ctx)
        alpha = T.dot(p_a,Ua) + ba
        alpha_shp = alpha.shape        
        alpha = self.tmp_softmax(alpha.reshape([alpha_shp[0],alpha_shp[1]]),mask_c) # softmax
        ctx_ = (context * alpha[:,:,None]).sum(0) # (m,pctx_dim)
        
        # selector
        if self.selector:
            sel_ = T.nnet.sigmoid(T.dot(h_tm1, self.Wsel) + self.bsel)
            sel_ = sel_.reshape([sel_.shape[0]])
            ctx_ = sel_[:,None] * ctx_        
        
        
        # end select        
        preact = T.dot(h_tm1, U)
        state_below = T.dot(x_t,W) + b 
        preact += state_below
        preact += T.dot(ctx_,Wc)
        
        i = T.nnet.sigmoid(self._slice(preact, 0, self.h_dim))
        f = T.nnet.sigmoid(self._slice(preact, 1, self.h_dim))
        o = T.nnet.sigmoid(self._slice(preact, 2, self.h_dim))
        c = T.tanh(self._slice(preact, 3, self.h_dim))
        
        c = f * c_tm1 + i * c
        c = m_t[:, None] * c + (1. - m_t)[:, None] * c_tm1

        h = o * T.tanh(c)
        h = m_t[:, None] * h + (1. - m_t)[:, None] * h_tm1
        
        return h, c
        
    def tmp_softmax(self,x,mask):
        exp_x = np.exp(-x)
        exp_x = exp_x*mask
        return exp_x/exp_x.sum(axis=0)
        
    def get_outputs(self,X,context = None,mask_x=None,mask_c = None,h0=None,c0=None): # need h0,c0, how to make it...
        """
        X: Nt * batch_size * dim
        mask: Nt * batch_size   
        """
        #assert mask is not None
        assert X.ndim==3 # need to modified to adaptive for 2-d input.

        Wpc = self.Wpc
        bc = self.bc
        
        n_samples = X.shape[1]

        if h0 is None:
            h_tm1 = T.unbroadcast(alloc_zeros_matrix(n_samples, self.h_dim), 1)
        else:
            h_tm1 = T.unbroadcast(h0,1)
        if c0 is None:
            c_tm1 = T.unbroadcast(alloc_zeros_matrix(n_samples, self.h_dim), 1)
        else:
            c_tm1 = T.unbroadcast(c0,1)
        
        p_ctx = T.dot(context,Wpc) + bc
        
        [h, c], updates = theano.scan(self._step,
                                      sequences=[mask_x, X],outputs_info=[h_tm1,c_tm1],non_sequences=[p_ctx,context,mask_c]
                                       )
        
        return h,c
        

    

    
        
