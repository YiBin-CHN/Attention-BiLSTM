#!/usr/bin/env
# -.- coding=utf-8 -.-
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
import json


class EvalCap:
    def __init__(self, ref, res): 
        """
        self.evalImgs = []
        
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': coco.getImgIds()}
        """
        self.eval = {}
        
        self.caps_ref = ref
        self.caps_res = res
        #print "%d references, %d results" %(len(caps_ref),len(caps_res))
        #print "Are reference names equal to result names?",set(caps_ref.keys())==set(caps_res.keys())


    def evaluate(self):
        """
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        ref = {} # reference
        res = {} # results
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)
        """
        # =================================================
        # Set up scorers
        # =================================================
        print 'setting up scorers...'
        scorers = [
            (Meteor(),"METEOR"),
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        s_res = {}
        for scorer, method in scorers:
            print 'computing %s score...'%(scorer.method())
            score, scores = scorer.compute_score(self.caps_ref, self.caps_res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    #self.setImgToEvalImgs(scs, imgIds, m)
                    print "%s: %0.3f"%(m, sc)
                    s_res[m] = sc
            else:
                self.setEval(score, method)
                #self.setImgToEvalImgs(scores, imgIds, method)
                print "%s: %0.3f"%(method, score)
                s_res[method] = score
        #self.setEvalImgs()
        return s_res

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]
        
