clc;
close all;
clear;
%% 读取数据
load('./data_problem4/data1/data1.mat');
%% 设置参数
A=data1;
[D,N]=size(A);
K=min(D,N);
tao=150;
f=zeros(K-1,1);
%% 实现算法
% miu=mean(A,2);
% [U,S,V]=svd(A-miu);
[U,S,V]=svd(A);
S_sort=diag(sort(diag(S),'DESCEND'));
for d=1:K-1
    sum=0;
    for i=d+1:K
        sum=sum+S_sort(i,i)^2;
    end
    if sum<=tao
        d_star=d;
        break
    end
end
%% 打印输出
figure(1)
hold on
plot(diag(S_sort).^2);
line([d_star,d_star],[0,S_sort(1,1)^2],'color','red','linestyle',':');
disp(d_star);