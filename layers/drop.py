#!/usr/bin/env
# -.- coding=utf-8 -.-
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T 

class drop(object):
    def __init__(self,p=0.5,rescale=True,deterministic=1.,independent=True):
        self.p = p
        self.rescale = rescale
        self.rng = RandomStreams(1234)
        self.deterministic = deterministic
        self.independent = independent
        
    def get_outputs(self,x):
        in_shape = x.shape
        p_retain = 1-self.p         
        u = self.rng.binomial(in_shape,p=p_retain,n=1,dtype=x.dtype)
        if self.rescale:
            d = u*x/p_retain
        else:
            d = u*x
        return T.switch(self.deterministic,x,d)