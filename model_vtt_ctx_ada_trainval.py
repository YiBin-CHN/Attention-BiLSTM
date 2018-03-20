import theano
import theano.tensor as T
from layers.lstm import LSTM
from layers.lstm_sa import LSTM_SA
from layers.drop import drop
import optimizers as optimizers
from layers.dense import Dense
from layers.embed import Embedding
from data.data_provider_vtt_trainval import DataProvider as data_provider
import time
import numpy as np
from utils.numpy_utils import numpyX
from theano.tensor import nnet as NN
import random
from utils.theano_utils import alloc_zeros_matrix,shared_zeros
from theano import function, config, shared, sandbox
import os,copy
from pycocoevalcap.test_metrics import EvalCap #maybe need to change...
global data_type
data_type = config.floatX

def l2(params):
    c = 0
    for p in params:
        c += (p**2).sum()
    return c


class Model(object):

    def __int__(self):
        pass

    def build_model(self, in_dim, h_dim, out_dim, model=None, weight_decay=0.01, optimizer='adadelta'):

        x_v = T.tensor3('x_v', dtype=data_type) # batch_size x dim
        x_s = T.tensor3('x_s',dtype=data_type) # Nt-1 x batch_size x dim
        y = T.tensor3('y', dtype=data_type) # Nt x batch_size x dim
        mask_v = T.matrix('mask_v', dtype=data_type) # Nt x batch_size mask_sent mask_video
        mask_s = T.matrix('mask_s', dtype=data_type)
        deterministic = T.scalar('deterministic',dtype=data_type)
        one = T.constant(1,dtype=data_type)
        zero = T.constant(0,dtype=data_type)
        lr = T.scalar('lr',dtype=data_type)
        mask_gen = T.matrix('mask_gen',dtype=data_type)
        mb_BOS = T.vector('mb_BOS',dtype=data_type)
        #maxlen = T.scalar('ml',dtype='int64')
        # layers
        l_lstm_f = LSTM(4096,h_dim)
        l_lstm_b = LSTM(4096,h_dim)
        l_lstm_v = LSTM_SA(h_dim*2,h_dim,4096,h_dim*2)
        l_word_em = Embedding(out_dim,in_dim)
        l_lstm_t = LSTM_SA(in_dim,h_dim,h_dim,h_dim)
        l_map = Dense(h_dim,out_dim)
        layers = [l_lstm_f,l_lstm_b,l_lstm_v,l_word_em,l_lstm_t,l_map]

        # drop_layers
        l_drop_xv = drop(0.2,deterministic=deterministic)
        l_drop_em = drop(0.2,deterministic=deterministic)
        l_drop_t = drop(0.5,deterministic=deterministic)
        l_drop_f = drop(0.5,deterministic=deterministic)
        l_drop_b = drop(0.5,deterministic=deterministic)
        x_v = l_drop_xv.get_outputs(x_v)



        # forward pass
        out_lstm_f,_ = l_lstm_f.get_outputs(x_v,mask_v)
        out_lstm_f = l_drop_f.get_outputs(out_lstm_f)
        out_lstm_b,_ = l_lstm_b.get_outputs(x_v[::-1],mask_v[::-1])
        out_lstm_b = l_drop_b.get_outputs(out_lstm_b)
        in_lstm_v = T.concatenate([out_lstm_f,out_lstm_b[::-1]],axis=2)
        out_lstm_v,c_v = l_lstm_v.get_outputs(in_lstm_v,context=x_v,mask_x=mask_v,mask_c=mask_v)
        out_word_em = l_word_em.get_outputs(x_s)
        out_word_em = l_drop_em.get_outputs(out_word_em)
        out_lstm_t,_ = l_lstm_t.get_outputs(out_word_em,mask_x=mask_s,context=out_lstm_v,mask_c=mask_v,h0=out_lstm_v[-1],c0=c_v[-1])
        out_lstm_t = l_drop_t.get_outputs(out_lstm_t)
        out_map = l_map.get_outputs(out_lstm_t)

        pred,_ = theano.scan(NN.softmax,sequences=out_map)

        # cost caculating
        cost_o,_ = theano.scan(NN.categorical_crossentropy,sequences=[pred,y])
        cost_o = cost_o*mask_s
        cost_o = cost_o.sum()/mask_s.sum()
        params_re = []
        for l in layers:
            params_re += l.regulariable
        cost_w = 0.5*weight_decay*l2(params_re)/mask_s.sum()
        cost = cost_o + cost_w

        self.params =  l_lstm_f.params + l_lstm_b.params + l_lstm_v.params + l_word_em.params + l_lstm_t.params + l_map.params

        p_ctx = T.dot(out_lstm_v,l_lstm_t.Wpc) + l_lstm_t.bc
        def _step(x_t,h_tm1,c_tm1,p_ctx,context):
            m_t = T.ones_like(x_t).astype(data_type)
            o_em_t = l_word_em.W[x_t.astype('int64')]
            h_t_t,c_t_t = l_lstm_t._step(m_t,o_em_t,h_tm1,c_tm1,p_ctx,context,mask_v)
            o_map_t = l_map._step(h_t_t)
            prob_p = NN.softmax(o_map_t)
            word_t = T.argmax(prob_p,axis=1).astype(data_type) # return an integer, the index of max value
            return word_t,h_t_t,c_t_t

        [words_idx,out_val_t,c_t],_ = theano.scan(_step,
                                   outputs_info=[dict(initial=mb_BOS),
                                                 dict(initial=out_lstm_v[-1]),
                                                 dict(initial=c_v[-1])],
                                   non_sequences=[p_ctx,out_lstm_v],
                                   n_steps=self.t_maxlen)

        val_model = theano.function(inputs=[x_v, mask_v,mb_BOS],
                                    givens={deterministic:one},
                                    outputs=words_idx)

        self.optimizer = optimizers.get(optimizer)
        grads = self.optimizer.get_gradients(cost, self.params)
        updates = self.optimizer.get_updates(cost, self.params)

        train_model = theano.function(inputs =[x_v, x_s, y, mask_v,mask_s],
                                      outputs=[cost,cost_o],updates=updates,givens={deterministic:zero}
                                      )

        return train_model, val_model

    def gen_sent(self,model,feat,mask_v,BOS_idx):
        """
        This method for generating sentence of any batch_size, which is corresponding to feat.
        """
        batch_size = mask_v.shape[1]
        t_BOS = np.zeros((batch_size,),dtype=data_type)
        t_BOS[:] = BOS_idx
        t_idxs = model(feat,mask_v,t_BOS)
        return t_idxs

    def fit(self,
            patience=20,
            batch_size=512,
            valid_batch_size=256,
            max_epochs=100000,
            lr=0.1,
            maxlen=55,
            t_maxlen=20,
            in_dim=1024, # lstm_t in_dim/word embedding dim
            h_dim = 512,##############
            dispFreq=10,
            valid_freq=20,
            saveFreq=20,
            out_path='models',
            test_path='test_results',
            savename='vtt_att_ctx'
            ):
        self.t_maxlen = t_maxlen
        if not os.path.isdir(out_path):
          os.mkdir(out_path)
        abs_savename = os.path.join(out_path,savename)
        t_name = os.path.join(test_path,savename)
        print 'loading data...'
        dp = data_provider()############change for need
        tr_num = dp.tr_num
        val_num = dp.val_num
        vocab_size = dp.vocab_size
        val_feat,mask_v,val_caps_ref = dp.get_val(497)
        ts_feat,mask_t,ts_caps_ref = dp.get_val(2990,mode='test')
        val = dp.val
        tr = dp.tr
        ts = dp.ts
        val_high_on = False
        val_start = False
        tr_iter = range(tr_num)
        ##############
        val_feat_num = len(val)
        ts_feat_num = len(ts)
        val_iter = range(val_feat_num)
        ts_iter = range(ts_feat_num)
        ####################
        print "Max len",maxlen
        print 'Building model...'
        train_model, valid_model = self.build_model(in_dim, h_dim, vocab_size, optimizer='adadelta')

        
        print 'Optimization...'
        f = open(abs_savename+'_log.txt','w')
        history_errs = []
        best_me = None
        best_b4 = None
        bad_count = 0
        lr_cnt = 0

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.time()
        toc = time.time()
        # valid_imgs = dp.get_batches(split = 'val')
        val_score = {}

        try:
            for eidx in range(max_epochs):
                n_samples = 0

                random.shuffle(tr_iter)
                for i in range(0,tr_num,batch_size):
                    tic = toc
                    batch_ids = tr_iter[i:min(i+batch_size,tr_num)]
                    t_v,t_s,t_y,t_mv,t_ms = dp.get_batch(batch_ids,maxlen=maxlen)

                    # add pad
                    uidx += 1
                    n_samples += 1
                    cost,cost_o = train_model(t_v,t_s,t_y,t_mv,t_ms)

                    if np.isnan(cost) or np.isinf(cost):
                        print 'bad cost detected: ', cost
                        estop = True
                        break
                        #return 1., 1., 1.

                    if np.mod(uidx, dispFreq) == 0:
                        toc = time.time()
                        print 'Epoch', eidx, 'Update', uidx,'lr',lr, 'Cost', cost,'cost_o',cost_o
                        print 'Time',toc-tic
                        print >>f,'Epoch', eidx, 'Update', uidx,'lr',lr, 'Cost', cost,'cost_o',cost_o

                    if not val_start and cost_o<3.8:
                        val_start = True

                    if val_start and np.mod(uidx, valid_freq) == 0:

                        val_caps_res = {}
                        
                        for val_i in range(0,val_feat_num,valid_batch_size):
                            val_stop = min(val_feat_num,valid_batch_size+val_i)
                            batch_val_feat = val_feat[:,val_i:val_stop,:]
                            batch_val_mask = mask_v[:,val_i:val_stop]
                            batch_val_idx = self.gen_sent(valid_model,batch_val_feat,batch_val_mask,dp.BOS)
                            if val_i == 0:
                                val_idxs = batch_val_idx
                            else:
                                val_idxs = np.concatenate([val_idxs,batch_val_idx],axis=1)
                        assert val_idxs.shape[1]==mask_v.shape[1]
                        print "val test true"
                        # val_idxs = valid_model(val_feat,mask_v,mb_BOS)
                        for samp_idx in range(val_idxs.shape[1]):
                            cap_l = []
                            for idx in val_idxs[:,samp_idx]:
                                if idx==dp.EOS:
                                    break
                                cap_l.append(dp.vocab_list[int(idx)])
                            val_caps_res[val[samp_idx]] = [' '.join(cap_l)]
                        # print samples
                        for p_idx in range(10):
                            iii = random.randint(0,len(val)-1)
                            print val[iii],val_caps_res[val[iii]]
                        # evaluate with metrics
                        caps_eval = EvalCap(val_caps_ref,val_caps_res)
                        s_res = caps_eval.evaluate()

                        if not val_high_on and (s_res["METEOR"]>0.304 or s_res["Bleu_4"]>0.42):
                            val_high_on = True
                            valid_freq = 10
                            saveFreq = 10
                            dispFreq = 10
                        for k,v in s_res.items():
                            print "%s : %4f" %(k,v)
                            if k not in val_score.keys():
                                val_score[k] = [v]
                            else:
                                val_score[k].append(v)
                        if len(val_score["METEOR"])>=2:
                            ME_sgn = s_res['METEOR']>=max(val_score["METEOR"][:-1])
                            B4_sgn = s_res['Bleu_4']>=max(val_score["Bleu_4"][:-1])
                        else:
                            ME_sgn = True
                            B4_sgn = True
                        if ME_sgn:
                            best_me = copy.deepcopy(self.params)

                            ts_caps_res = {}
                            for ts_i in range(0,ts_feat_num,valid_batch_size):
                                ts_stop = min(ts_feat_num,valid_batch_size+ts_i)
                                batch_ts_feat = ts_feat[:,ts_i:ts_stop,:]
                                batch_ts_mask = mask_t[:,ts_i:ts_stop]
                                batch_ts_idx = self.gen_sent(valid_model,batch_ts_feat,batch_ts_mask,dp.BOS)
                                if ts_i == 0:
                                    ts_idxs = batch_ts_idx
                                else:
                                    ts_idxs = np.concatenate([ts_idxs,batch_ts_idx],axis=1)
                            assert ts_idxs.shape[1]==mask_t.shape[1]
                            print "ts test true"                            
                            # ts_idxs = valid_model(ts_feat,mask_t,t_BOS)
                            for samp_idx in range(ts_idxs.shape[1]):
                                cap_l = []
                                for idx in ts_idxs[:,samp_idx]:
                                    if idx==dp.EOS:
                                        break
                                    cap_l.append(dp.vocab_list[int(idx)])
                                ts_caps_res[ts[samp_idx]] = [' '.join(cap_l)+' .']

                            # evaluate with metrics
                            caps_eval = EvalCap(ts_caps_ref,ts_caps_res)
                            s_res = caps_eval.evaluate()
                            # save samples
                            f_tme = open(t_name+'_ME.txt','w')
                            print >>f_tme,'============Score============\n'
                            for k,v in s_res.items():
                                print >>f_tme,k,v 
                            print >>f_tme,'===========Caption===========\n'                            
                            for n,c in ts_caps_res.items():
                                print >>f_tme,n,c
                            f_tme.close()                            
                            
                            bad_count = 0
                            lr_cnt = 0
                        if B4_sgn:
                            best_b4 = copy.deepcopy(self.params)
                            ts_caps_res = {}  
                            for ts_i in range(0,ts_feat_num,valid_batch_size):
                                ts_stop = min(ts_feat_num,valid_batch_size+ts_i)
                                batch_ts_feat = ts_feat[:,ts_i:ts_stop,:]
                                batch_ts_mask = mask_t[:,ts_i:ts_stop]
                                batch_ts_idx = self.gen_sent(valid_model,batch_ts_feat,batch_ts_mask,dp.BOS)
                                if ts_i == 0:
                                    ts_idxs = batch_ts_idx
                                else:
                                    ts_idxs = np.concatenate([ts_idxs,batch_ts_idx],axis=1)
                            assert ts_idxs.shape[1]==mask_t.shape[1]
                            print "ts test true"                             
                            # ts_idxs = valid_model(ts_feat,mask_t,t_BOS)
                            for samp_idx in range(ts_idxs.shape[1]):
                                cap_l = []
                                for idx in ts_idxs[:,samp_idx]:
                                    if idx==dp.EOS:
                                        break
                                    cap_l.append(dp.vocab_list[int(idx)])
                                ts_caps_res[ts[samp_idx]] = [' '.join(cap_l)+' .']
                            # evaluate with metrics
                            caps_eval = EvalCap(ts_caps_ref,ts_caps_res)
                            s_res = caps_eval.evaluate()                            
                            # save samples
                            f_tb4 = open(t_name+'_B4.txt','w')
                            print >>f_tb4,'============Score============\n'                            
                            for k,v in s_res.items():
                                print >>f_tb4,k,v 
                            print >>f_tb4,'===========Caption===========\n'                            
                            for n,c in ts_caps_res.items():
                                print >>f_tb4,n,c
                            f_tb4.close()
                            
                            bad_count = 0
                            lr_cnt = 0
                        if not ME_sgn and not B4_sgn:
                            bad_count += 1
                            lr_cnt += 1
                            if lr_cnt >5 and lr>0.0001:
                                lr = max(lr/2.0,0.0001)
                                print >>f,'\nLEARNING RATE DECAY',lr,'\n'
                                lr_cnt = 0
                            if bad_count > patience:
                                print 'Early Stop!'
                                estop = True
                                break

                    if abs_savename and np.mod(uidx, saveFreq) == 0:
                        if best_me is not None and best_b4 is not None:
                            print 'Saving...',
                            np.savez(abs_savename+'_ME_model.npz', *best_me)
                            np.savez(abs_savename+'_B4_model.npz', *best_b4)
                            print 'Done'

                print 'Seen %d samples' % n_samples
                if estop:
                    break

        except KeyboardInterrupt:
            print 'Training interrupt'
            print >>f,'Training interrupt'

        f.close()
        end_time = time.time()



if __name__ == '__main__':
    model = Model()
    model.fit( max_epochs=500, maxlen=55)


