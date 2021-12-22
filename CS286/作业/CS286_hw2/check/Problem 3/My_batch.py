# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 18:02:01 2020

@author: aiwan
"""
import random

def My_batch(inputlist1,inputlist2,batchsize):
    inputlist=[tuple(inputlist1[i],inputlist2[i]) for i in range(len(inputlist1))]
    result=[]
    input_list=inputlist.copy()
    while(len(input_list)>=batchsize):
        random.shuffle(input_list)
        result.append(input_list[0:batchsize])
        input_list=input_list[batchsize:]
    if input_list==[]:
        return result
    else:
        result.append(input_list)
        return result
