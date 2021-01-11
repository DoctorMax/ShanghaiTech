clc;
close all;
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
%% 算法实现
[x1J,k1J,loss1J]=jacobi_it(A1,b1,x1,x10,N,eps);
[x2J,k2J,loss2J]=jacobi_it(A2,b2,x2,x20,N,eps);
[x1G,k1G,loss1G]=gauss_it(A1,b1,x1,x10,N,eps);
[x2G,k2G,loss2G]=gauss_it(A2,b2,x2,x20,N,eps);
[x1S,k1S,loss1S]=SOR_it(A1,b1,x1,x10,N,eps,0.9);
[x2S,k2S,loss2S]=SOR_it(A2,b2,x2,x20,N,eps,0.9);
%% 打印输出
figure(1)
hold on
plot((1:k1J),loss1J,'r');
plot((1:k1G),loss1G,'g');
plot((1:k1S),loss1S,'b');
figure(2)
hold on
xlim([0,10])
plot((1:k2J),loss2J,'r');
plot((1:k2G),loss2G,'g');
plot((1:k2S),loss2S,'b');