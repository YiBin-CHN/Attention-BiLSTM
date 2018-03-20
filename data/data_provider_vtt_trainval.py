import os
import json
import scipy.io as sio
from collections import defaultdict
import time
from theano import function, config, shared
import numpy as np
from numpy import tile
import theano
import random

data_type = config.floatX
def sentence2vec(idx,dim,maxlen=100): #
    """
    idx is a list with words index.
    dim is word vector dimension, typically vocab_size
    """
    Nw = len(idx)
    wordvecs = np.zeros((maxlen,dim))
    if Nw<maxlen:
        for i in range(Nw):
            wordvecs[i][idx[i]] = 1
    else:
        for i in range(maxlen):
            wordvecs[i][idx[i]] = 1
    return wordvecs

class DataProvider(object):

    def __init__(self):
        """
        dataset: folder name in root. revised by Yi. 2016/02/29
        """
        ###########
        # prepare path and file
        list_path = '/mnt/disk1/msr_vtt/features'
        list_file = 'keyframe_samp30.txt'
        feat_path = '/mnt/disk1/msr_vtt/features'
        feat_file = 'keyframe_samp30_fc7_sorted.json'

        out_path = '/mnt/disk1/msr_vtt/features'

        tr_id = 6512
        val_id = 7009
        ts_id = 9999

        cap_file = '/mnt/disk1/msr_vtt/annotations/vtt_caps_idx.json'
        ########################
        tr = []
        val = []
        ts = []

        ##############
        # mapping video frame to feature index
        list_abs_path = os.path.join(list_path,list_file)

        self.name_map,self.out_of_order = self.get_feature_idx(list_abs_path)
        
       
        # tr_name_map is dict with video name as keys, and each value is list of idx of each frame feature.
        # end mapping.
        ################
        
        ###############
        f_caps = open(cap_file) ########
        jfcap = json.load(f_caps)
        sentences = jfcap['sentences'] ##########list with elements dict,{name,sentence}
        f_caps.close()
        vocab = jfcap['vocab_dict'] # type: dict, key:words value:indices
        vocab_size = len(vocab) # used to initiate 'We'.
        self.vocab_list = jfcap['vocab_list']
        ##############
        # split sentences to tr val ts 
        caps_tr = []
        caps_val = []
        caps_ts = []
        tr = set()
        val = set()
        ts = set()
        caps_name = set()
        for s in sentences:
            vid = s['video_id']
            caps_name.add(s['video_id'])
            vid = vid[5:]
            if int(vid)<=tr_id:
                caps_tr.append(s)
                tr.add(s['video_id'])
            elif int(vid)<=val_id:
                caps_val.append(s)
                val.add(s['video_id'])
            elif int(vid)<=ts_id:
                caps_ts.append(s)
                ts.add(s['video_id'])
            else:
                print "No such video ID"
                
        tr_num = len(caps_tr)
        val_num = len(caps_val)
        ts_num = len(caps_ts)
        print "tr,val,ts sen_num",tr_num,val_num,ts_num
        
        ###################
        # load features
        f = open(os.path.join(feat_path,feat_file))
        jd = json.load(f)
        f.close()
        fc7 = jd['fc7']
        feat_all = np.asarray(fc7)
        print 'features loaded, features shape',feat_all.shape
        
        ## verify
        print "name verify",caps_name==set(self.name_map.keys())
        

        self.tr = list(tr)
        self.val = list(val)
        self.ts = list(ts)
        #self.vocab_list = vocab_list
        self.vocab_size = vocab_size
        self.caps_tr = caps_tr
        self.caps_val = caps_val
        self.caps_ts = caps_ts
        self.feat = feat_all
        #self.name_feat_idx = name_feat_idx
        self.tr_num = tr_num
        self.val_num = val_num
        self.ts_num = ts_num
        self.out_path = out_path
        self.BOS = vocab['#BOS#']
        self.EOS = vocab['.']
        print 'All data ready!'
        print 'Here are %d tr and %d val captions in total!' %(tr_num,val_num)
        print "vocabulary size:",vocab_size
        print "verify name index and feature order, should display\
                [0:9],[9972:10000],[44997:45008],[99993:100009],[123448:123459],[144384:]"
        print self.name_map['video6881']
        print self.name_map['video7890']
        print self.name_map['video6172']
        print self.name_map['video7735']
        print self.name_map['video3606']
        print self.name_map['video9575']
    def get_feature_idx(self,listfile):
        """
        listfile should be an absolute path
        """
        flist = open(listfile)
        list_name = os.path.basename(listfile)
        list_name = list_name.split('.txt')[0]
        i = 0
        video_cnt = 0
        name_feat_idx = {}
        curr_video = ''
        out_of_order = {}
        for line in flist:
            line = line.strip()
            line = line.split()[0]
            dir_name,frame_name = os.path.split(line)
            video_name = os.path.basename(dir_name)
            if video_name!=curr_video:
                curr_video=video_name
                if curr_video not in name_feat_idx.keys():
                    name_feat_idx[curr_video]=[]
                    video_cnt += 1
                else:
                    out_of_order[str(i)]=video_name
            name_feat_idx[curr_video].append(i)
            i += 1
        flist.close()
        if len(out_of_order)>0:
            f_out_of_order = open(os.path.join(self.out_path,list_name+'out_of_order.txt'),'w')
            for idx,v in out_of_order:
                print >>f_out_of_order,v,idx # separated by space
            f_out_of_order.close()
        else:
            print "No out of order images in %s" %list_name
        print '%s: %d videos, %d frames' %(list_name,video_cnt,i)
        return name_feat_idx,out_of_order



    def get_batch(self,batch_idx,mode='train', maxlen=80):
        """
        input
            batch_idx: list of caps idx. integer..
        return
            mask: Nt x batch_size, may be truncated by longest sentence, without EOS
            features: batch_size x dim_f // Nt(img=1) * batch_size * dim_f
            sentences: Nt-1 x batch_size x dim_w (not embed), without EOS
            Y: Nt x batch_size x dim_w, with EOS and BOS
        """
        if mode== 'train':
            caps = self.caps_tr
        elif mode=='valid':
            caps = self.caps_val
        else:
            raise ValueError('No such mode')
        name_map = self.name_map
        feat = self.feat
        
        mask_sent = np.zeros((maxlen,len(batch_idx))).astype(data_type)
        mask_video = np.zeros((maxlen,len(batch_idx))).astype(data_type)
        mask_video_rev = np.zeros((maxlen,len(batch_idx))).astype(data_type)
        features = np.zeros((maxlen,len(batch_idx),4096)).astype(data_type)
        #features_rev = np.zeros((maxlen,len(batch_idx),4096)).astype(data_type)
        Y = np.zeros((maxlen,len(batch_idx),self.vocab_size)).astype(data_type)
        for i in range(len(batch_idx)):
            idx = batch_idx[i]
            samp = caps[idx]
            name = samp['video_id'] # withou extension
            idx_sen = samp['idx']
            wordvecs = sentence2vec(idx_sen,self.vocab_size,maxlen=maxlen) # Xs
            len_sent = len(idx_sen)
            if len_sent<maxlen:
                Y[:,i,:] = wordvecs
                mask_sent[:len_sent-1,i]=1 #  contain BOS no EOS
            else:
                Y[:,i,:] = wordvecs[:maxlen,:]
                mask_sent[:-1,i]=1
            feat_idxs = name_map[name]
            len_feat = len(feat_idxs)
            if len_feat>maxlen:
                stop_idx = maxlen
            else:
                stop_idx = len_feat
            mask_video[:stop_idx,i] = 1
            for feat_idx in range(stop_idx):
                features[feat_idx,i,:] = feat[feat_idxs[feat_idx]]
                #features_rev[feat_idx,i,:] = feat[feat_idxs_rev[feat_idx]]

        mask_video_sum = mask_video.sum(axis=1)
        try:
            len_videos = list(mask_video_sum).index(0)  # 1 for idx from 0, discard +1, revised by Yi, 20160926
        except ValueError:
            len_videos = maxlen
        mask_sent_sum = mask_sent.sum(axis=1)
        try:
            len_sents = list(mask_sent_sum).index(0) # discard +2, see line 184-187.
        except ValueError:
            len_sents = maxlen - 1
        if len_sents+len_videos>maxlen:
            if len_sents>=26: # 85528 in 85550 sentences are less than 40 words.
                mask_sent = mask_sent[:26-1,:]
                mask_video = mask_video[:maxlen-26+1,:]  # satisfy sentence priority
                features = features[:maxlen-26+1,:]
                sentences = Y[:26-1,:]
                Y = Y[1:26,:]
            else:
                mask_sent = mask_sent[:len_sents-1,:]
                mask_video = mask_video[:maxlen-len_sents+1,:]
                features = features[:maxlen-len_sents+1,:]
                sentences = Y[:len_sents-1,:]
                Y = Y[1:len_sents,:]
        else:
            mask_video = mask_video[:len_videos,:]
            mask_sent = mask_sent[:len_sents-1,:]
            features = features[:len_videos,:]
            sentences = Y[:len_sents-1,:]
            Y = Y[1:len_sents,:]

        return features,sentences,Y,mask_video,mask_sent

    def get_val(self,batch_size,mode='val',maxlen=50): # maxlen indicate that number of video frames
        if mode == 'val':
            caps = self.caps_val
            name_set = self.val

        elif mode == 'test':
            caps = self.caps_ts
            name_set = self.ts

        else:
            raise ValueError("No such mode (%s)!" %mode)
        #######
        name_map = self.name_map
        feat = self.feat
        #caps = self.caps_val
        caps_ref_dict = {}

        for samp in caps:
            name = samp['video_id']
            idxs = samp['idx']
            sent = []
            for i in idxs[1:-1]: # no EOS
                sent.append(self.vocab_list[i])
            cap = ' '.join(sent)+' .'

            if name not in caps_ref_dict.keys():
                caps_ref_dict[name] = [cap]

            else:
                caps_ref_dict[name].append(cap)

        caps_ref = caps_ref_dict
        # verify keys...
        print "%d reference in %s" %(len(caps_ref),mode)
        print "Are names equal?",set(caps_ref.keys())==set(name_set)


        features = np.zeros((maxlen,batch_size,4096)).astype(data_type)
        mask = np.zeros((maxlen,batch_size)).astype(data_type)
        i = 0
        for name_f in name_set:
            feat_idxs = name_map[name_f]
            len_feat = len(feat_idxs)
            if len_feat <= maxlen:
                for feat_idx in range(len_feat):
                    features[feat_idx,i,:] = feat[feat_idxs[feat_idx]]
                    mask[feat_idx,i] = 1
            else:
                for feat_idx in range(maxlen):
                    features[feat_idx,i,:] = feat[feat_idxs[feat_idx]]
                    mask[:,i] = 1 
            i += 1
        return features,mask,caps_ref



