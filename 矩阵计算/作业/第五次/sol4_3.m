clc;
close all;
clear;
%% 读取数据
load('./data_problem4/all_data.mat');
%% 设置参数
X=data_train;
X_test=data_test;
num=size(X,2);
d=8;
label=zeros(20,1);
%% 算法实现
miu=mean(X,2);
[U,S,V]=svd(X-miu);
[S_sort,index]=sort(diag(S),'DESCEND');
S_sort_ex=diag(S_sort);
S_sort_ex(num:size(X,1),:)=0;
U_sort=U(:,index);
U_d=U_sort(:,1:d);
miuface=reshape(miu,48,42);
for i=1:d
    temp=U_d(:,i);
    U_d_norml(:,i)=(temp-min(temp))/(max(temp)-min(temp));
end
Y_test=U_d'*(X_test-miu);
Y_train=U_d'*(X-miu);
X_test_proj=miu+U_d*Y_test;
X_train_proj=miu+U_d*Y_train;
for i=1:20
    loss=zeros(40,1);
    for j=1:40
        loss(j)=norm(X_test_proj(:,i)-X_train_proj(:,j),2);
    end
    [~,index]=min(loss);
    label(i)=ceil(index/10);
end
%% 打印输出
error=sum(Y_label_test~=label');
error_rate=error/20;