# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:48:26 2020

@author: aiwan
"""
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

##########################################################
'''读入数据并且去掉位置'''
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.drop(['POS'], axis =1)
data = data.fillna(method="ffill")

##########################################################


##########################################################
'''将单词全部存放到字典里，每个单词对应的值表示第几个单词'''
word_to_ix = {}
words = set(list(data['Word'].values))
for w in words:
    word_to_ix[w]=len(word_to_ix)

##########################################################



##########################################################
'''把17个tag分别对应17个数'''
tag_dicts={}
tags = set(list(data["Tag"].values))
for t in tags:
    tag_dicts[t]=len(tag_dicts)
n_tags = len(tags)


##########################################################



##########################################################
'''把47959个句子里的每一个词进行分类然后用list套list的形式存在
sentences里面,此时类别仍然为字符串类型而不是前面的数字类型
'''
agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),s["Tag"].values.tolist())]
grouped = data.groupby("Sentence #").apply(agg_func)
sentences = [s for s in grouped]

##########################################################


##########################################################
'''将每一句话用数字的形式存在new_data里，对应位置的标签存在
new_tag里并且制作完训练集测试集和验证集'''
max_len = 50
X = [[w[0]for w in s] for s in sentences]
Y = [[w[1]for w in s] for s in sentences]
new_data = []
new_tags=[]
for seq,tag in zip(X,Y):
    new_seq=[]
    new_tag=[]
    for i in range(max_len):
        try:
            new_seq.append(word_to_ix[seq[i]])
            new_tag.append(tag_dicts[tag[i]])
        except:
            pass
    new_data.append(new_seq)
    new_tags.append(new_tag)
    
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, new_tags, test_size=0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3)

##########################################################



##########################################################
'''用来制作自己的batch'''
def My_batch(inputlist1,inputlist2,batchsize):
    inputlist=[[inputlist1[i],inputlist2[i]] for i in range(len(inputlist1))]
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
##########################################################








torch.manual_seed(1)

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq




START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
train_losslist=[]
valid_losslist=[]


tag_dicts['<START>']=17
tag_dicts['<STOP>']=18



model = BiLSTM_CRF(len(word_to_ix), tag_dicts, EMBEDDING_DIM, HIDDEN_DIM)


optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


training_data=My_batch(X_train[:10000],y_train[:10000],32)
valid_data=My_batch(X_valid,y_valid,32)
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10): # again, normally you would NOT do 300 epochs, it is toy data
    for i in tqdm(range(len(training_data))):
        batch=training_data[i]

        for j in range(len(batch)):
            sample=batch[j]
            sentence=sample[0]
            tags=sample[1]
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

        # Step 2. Get our inputs ready for the ne
            sentence_in = torch.tensor(sentence,dtype=torch.long)
            targets = torch.tensor(tags, dtype=torch.long)

        # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)
            
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
            loss.backward()
            optimizer.step()
        train_losslist.append(loss)  
torch.save(model,'BiLSTM_CRF.pkl')


model=torch.load('BiLSTM_CRF.pkl')



f1_loss=[]

for i in tqdm(range(len(valid_data))):
    loss_tmp=torch.zeros(32)
    f1_tmp=torch.zeros(32)
    for j in range(len(valid_data[i])):

        sample=valid_data[i][j]
        predict=model(torch.tensor(sample[0],dtype=torch.long))
        loss=model.neg_log_likelihood(torch.tensor(sample[0],dtype=torch.long),
                                 torch.tensor(sample[1],dtype=torch.long))
        loss_tmp[j]=loss
        predict_np=np.array(predict[1])
        y_valid_np=np.array(sample[1])
        f1score=f1_score(y_valid_np, predict_np,average='micro')
        f1_tmp[j]=f1score
    average0=torch.mean(loss_tmp)
    average1=torch.mean(f1_tmp)
    valid_losslist.append(average0)
    f1_loss.append(average1)

figure1=plt.figure(figsize=(10,10))
plt.plot(train_losslist,'b',valid_losslist,'r')
plt.show()
figure2=plt.figure(figsize=(10,10))
plt.plot(f1_loss,'r')
plt.show()



#with torch.no_grad():
    #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#    print(model(precheck_sent))
# Check predictions after training

















