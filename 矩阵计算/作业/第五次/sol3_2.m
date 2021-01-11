close all;
clc;
clear;
%% 读取数据
A1=importdata('./data_problem3/A1.txt');
A2=importdata('./data_problem3/A2.txt');
b1=importdata('./data_problem3/b1.txt');
b2=importdata('./data_problem3/b2.txt');
x1=importdata('./data_problem3/x1.txt');
x2=importdata('./data_problem3/x2.txt');
%% 设置参数
N=1000;
eps=1e-6;
x10=2*(rand(size(x1))-0.5);
x20=2*(rand(size(x2))-0.5);
k1List=[];
k2List=[];
%% 算法实现
for w=0:0.1:1
    [x1S,k1,~]=SOR_it(A1,b1,x1,x10,N,eps,w);
    [x2S,k2,~]=SOR_it(A2,b2,x2,x20,N,eps,w);
    k1List=[k1List,k1];
    k2List=[k2List,k2];
end
%% 打印输出
figure(1)
plot((0:0.1:1),k1List);
hold on
plot((0:0.1:1),k2List);