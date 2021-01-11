clc;
clear;
%% ��ȡ����
data=importdata('./data_problem1/data.txt');
label=importdata('./data_problem1/label.txt');
%% ���ò���
[m,n]=size(data);
lambda=0.1;
c=1e5;
A=data;
y=label;
x=2*(rand(n,1)-0.5);
k=6000; % ������������
l1List=zeros(k,1);
%% ���е���
for i=1:k
    x=soft(A'/c*(y-A*x)+x,lambda/c);
    l1List(i)=l1_regu(A,x,y,lambda); % �۲�loss�仯
end
%% ��ӡ���
figure(1)
plot(l1List);
L0=sum(x(:)~=0);
loss=norm(A*x-y,2)^2;
save ./sol1.txt -ascii x