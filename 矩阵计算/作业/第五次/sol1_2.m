clc;
clear;
%% 读取数据
data=importdata('./data_problem1/data.txt');
label=importdata('./data_problem1/label.txt');
%% 设置参数
[m,n]=size(data);
lambda=0.1;
gamma=1e-5;
c=1e5;
A=data;
y=label;
x=2*(rand(n,1)-0.5);
k=6000; % 迭代次数上限
l2List=zeros(k,1);
%% 进行迭代
for i=1:k
    x=x-gamma*2*(A'*A*x-A'*y+lambda*x);
    l2List(i)=l2_regu(A,x,y,lambda); % 观察loss变化
end
%% 打印输出
figure(1)
plot(l2List);
L0=sum(x(:)~=0);
loss=norm(A*x-y,2)^2;
save ./sol2.txt -ascii x