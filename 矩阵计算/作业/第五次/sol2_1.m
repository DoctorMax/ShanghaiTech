clc;
clear;
format rat 
%% 参数设置
A=[0,1,0,0;0,0,2,0;0,0,0,3;1/6000,0,0,0];
%% 算法实现
s= svd(A)
e= eig(A)