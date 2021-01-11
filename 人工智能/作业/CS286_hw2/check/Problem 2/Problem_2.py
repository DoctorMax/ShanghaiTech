# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:37:28 2020

@author: aiwan
"""
import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader
import time

import os
import sys
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch import optim
from tqdm import tqdm




torch.manual_seed(1)
EPOCH = 200
BATCH_SIZE = 128
LR = 1e-4
N_TEST_IMG = 5


dataset = np.load('counts.npy')
labels = np.loadtxt("labels.txt")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def return_out_mask(batch_data):
    '''
    Input: NxM np.array of counts data
    Output: NxM np.array filled with 1 corresponding to non-zero inputs
            and 0 corresponding to dropped inputs
    '''
    batch_data_numpy = batch_data.numpy()
    zero_indices = np.nonzero(batch_data_numpy == 0)
    mask = np.ones_like(batch_data_numpy)
    zeros = np.zeros_like(batch_data_numpy)
    mask[zero_indices] = zeros[zero_indices] # Fills in 0's into appropriate indices
    mask_torch  = torch.from_numpy(mask)
    return mask_torch

class AutoEncoder(nn.Module):
    def __init__(self,embed):
        super(AutoEncoder,self).__init__()
        self.encoder  =  nn.Sequential(
            nn.Linear(1000,500),
            nn.Tanh(),
            nn.Linear(500,250),
            nn.Tanh(),   
            nn.Linear(250,embed),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embed,250),
            nn.Tanh(),
            nn.Linear(250,500),
            nn.Tanh(),
            nn.Linear(500,1000),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded).mul(return_out_mask(x))
        return encoded, decoded

embedlist=[5,10,50,100]


Coder_5 = AutoEncoder(embedlist[0])
Coder_10 = AutoEncoder(embedlist[1])
Coder_50 = AutoEncoder(embedlist[2])
Coder_100 = AutoEncoder(embedlist[3])

optimizer_5 = torch.optim.Adam(Coder_5.parameters(),lr=LR)
optimizer_10 = torch.optim.Adam(Coder_10.parameters(),lr=LR)
optimizer_50 = torch.optim.Adam(Coder_50.parameters(),lr=LR)
optimizer_100 = torch.optim.Adam(Coder_100.parameters(),lr=LR)

loss_func = nn.MSELoss()


for epoch in tqdm(range(EPOCH)):
    for step, x in enumerate(dataloader):
        b_x_5 = x.view(-1,1000)
        encoded_5, decoded_5 = Coder_5(b_x_5)
        loss_5 = loss_func(decoded_5, b_x_5)
        optimizer_5.zero_grad()
        loss_5.backward()
        optimizer_5.step()
        
        b_x_10 = x.view(-1,1000)
        encoded_10, decoded_10 = Coder_10(b_x_10)
        loss_10 = loss_func(decoded_10, b_x_10)
        optimizer_10.zero_grad()
        loss_10.backward()
        optimizer_10.step()
        
        
        b_x_50 = x.view(-1,1000)
        encoded_50, decoded_50 = Coder_50(b_x_50)
        loss_50 = loss_func(decoded_50, b_x_50)
        optimizer_50.zero_grad()
        loss_50.backward()
        optimizer_50.step()
        
        
        b_x_100 = x.view(-1,1000)
        encoded_100, decoded_100 = Coder_100(b_x_100)
        loss_100 = loss_func(decoded_100, b_x_100)
        optimizer_100.zero_grad()
        loss_100.backward()
        optimizer_100.step()
        
        

torch.save(Coder_5,'AutoEncoder_5.pkl')
torch.save(Coder_10,'AutoEncoder_10.pkl')
torch.save(Coder_50,'AutoEncoder_50.pkl')
torch.save(Coder_100,'AutoEncoder_100.pkl')
print('________________________________________')
print('finish training')



Coder_5 = torch.load('AutoEncoder_5.pkl')
Coder_10 = torch.load('AutoEncoder_10.pkl')
Coder_50 = torch.load('AutoEncoder_50.pkl')
Coder_100 = torch.load('AutoEncoder_100.pkl')


# T-SNE & PCA Plot of Counts Data
tsne_data = TSNE(n_components=2).fit_transform(dataset)
plt.scatter(tsne_data[:,0],tsne_data[:,1],c=labels,s=3)
plt.title('data_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(dataset)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('data_pca')
plt.show()




# T-SNE & PCA Plot of Encodings
#####################################################
dataset_tensor=torch.from_numpy(dataset)

encoded_data_5, decoded_data_5 =Coder_5(dataset_tensor)
encoded_data_10, decoded_data_10 =Coder_10(dataset_tensor)
encoded_data_50, decoded_data_50 =Coder_50(dataset_tensor)
encoded_data_100, decoded_data_100 =Coder_100(dataset_tensor)
# np.array
encoded_data_5_numpy=encoded_data_5.detach().numpy()
encoded_data_10_numpy=encoded_data_10.detach().numpy()
encoded_data_50_numpy=encoded_data_50.detach().numpy()
encoded_data_100_numpy=encoded_data_100.detach().numpy()

decoded_data_5_numpy=decoded_data_5.detach().numpy()
decoded_data_10_numpy=decoded_data_10.detach().numpy()
decoded_data_50_numpy=decoded_data_50.detach().numpy()
decoded_data_100_numpy=decoded_data_100.detach().numpy()
#####################################################

# T-SNE & PCA Plot of Counts Data
tsne_data = TSNE(n_components=2).fit_transform(dataset)
plt.scatter(tsne_data[:,0],tsne_data[:,1],c=labels,s=3)
plt.title('data_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(dataset)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('data_pca')
plt.show()
#####################################################




tsne_latent = TSNE(n_components=2).fit_transform(encoded_data_5_numpy)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('encoded_5_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(encoded_data_5_numpy)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('encoded_5_pca')
plt.show()


######################################################

#####################################################




tsne_latent = TSNE(n_components=2).fit_transform(encoded_data_10_numpy)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('encoded_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(encoded_data_10_numpy)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('encoded_pca')
plt.show()


######################################################

#####################################################




tsne_latent = TSNE(n_components=2).fit_transform(encoded_data_50_numpy)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('encoded_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(encoded_data_50_numpy)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('encoded_pca')
plt.show()


######################################################

#####################################################




tsne_latent = TSNE(n_components=2).fit_transform(encoded_data_100_numpy)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('encoded_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(encoded_data_100_numpy)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('encoded_pca')
plt.show()





# T-SNE & PCA Plot of Reconstructions
#####################################################
reconstructions= decoded_data_5_numpy
#####################################################
tsne_latent = TSNE(n_components=2).fit_transform(reconstructions)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('reconstructed_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(reconstructions)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('reconstructed_pca')
plt.show()

# T-SNE & PCA Plot of Reconstructions
#####################################################
reconstructions= decoded_data_10_numpy
#####################################################
tsne_latent = TSNE(n_components=2).fit_transform(reconstructions)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('reconstructed_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(reconstructions)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('reconstructed_pca')
plt.show()
# T-SNE & PCA Plot of Reconstructions
#####################################################

reconstructions= decoded_data_50_numpy
#####################################################
tsne_latent = TSNE(n_components=2).fit_transform(reconstructions)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('reconstructed_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(reconstructions)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('reconstructed_pca')
plt.show()

# T-SNE & PCA Plot of Reconstructions
#####################################################
reconstructions= decoded_data_100_numpy
#####################################################
tsne_latent = TSNE(n_components=2).fit_transform(reconstructions)
plt.scatter(tsne_latent[:,0],tsne_latent[:,1],c=labels,s=3)
plt.title('reconstructed_tsne')
plt.show()

pca = PCA(n_components=2)
pca_data = pca.fit_transform(reconstructions)
plt.scatter(pca_data[:,0],pca_data[:,1],c=labels,s=3)
plt.title('reconstructed_pca')
plt.show()




















'''
#数据的空间形式的表示
view_data = train_data.train_data[:200].view(-1, 28*28).type(torch.FloatTensor)/255.
encoded_data, _ = Coder(view_data)    # 提取压缩的特征值
fig = plt.figure(2)
ax = Axes3D(fig)    # 3D 图
# x, y, z 的数据值
X = encoded_data.data[:, 0].numpy()
Y = encoded_data.data[:, 1].numpy()
Z = encoded_data.data[:, 2].numpy()
# print(X[0],Y[0],Z[0])
values = train_data.train_labels[:200].numpy()  # 标签值
for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9))    # 上色
    ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()
'''

#原数据和生成数据的比较
'''
plt.ion()
plt.show()


for i in range(10):
    test_data = train_data.train_data[i].view(-1,28*28).type(torch.FloatTensor)/255.
    _,result = Coder(test_data)

    # print('输入的数据的维度', train_data.train_data[i].size())
    # print('输出的结果的维度',result.size())
    
    im_result = result.view(28,28)
    # print(im_result.size())
    plt.figure(1, figsize=(10, 3))
    plt.subplot(121)
    plt.title('test_data')
    plt.imshow(train_data.train_data[i].numpy(),cmap='Greys')

    plt.figure(1, figsize=(10, 4))
    plt.subplot(122)
    plt.title('result_data')
    plt.imshow(im_result.detach().numpy(), cmap='Greys')
    plt.show()
    plt.pause(0.5)

plt.ioff()
'''