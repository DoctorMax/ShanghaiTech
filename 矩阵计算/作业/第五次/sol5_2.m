clc;
close all;
clear;
%% 读取数据
load('./data_problem4/data1/data1.mat');
%% 设置参数
A=data1;
[D,N]=size(A);
K=min(D,N);
tao=150;%0.1,0.05,0.02,0.005
f=zeros(K-1,1);
%% 实现算法
% miu=mean(A,2);
% [U,S,V]=svd(A-miu);
[U,S,V]=svd(A);
S_sort=diag(sort(diag(S),'DESCEND'));
for d=1:K-1
    sum1=0;
    sum2=0;
    for i=d+1:K
        sum1=sum1+S_sort(i,i)^2;
    end
    for i=1:K
        sum2=sum2+S_sort(i,i)^2;
    end
    f(d)=sum1/sum2;
%     if f(d)<=tao
%         d_star=d;
%         break
%     end
end
% compre_rate=d_star*(D+N+1)/(D*N);
%% 打印输出
figure(1)
stairs(f);
% disp(compre_rate);