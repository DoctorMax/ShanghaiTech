clc;
clear;
%%
a=66;
b=77;
A=[0,a;b,0];
kmax=100;%设置最大迭代次数，防止不收敛时无限循环
q=zeros(2,kmax);%这样子方便点
z=zeros(2,kmax);
lambda=zeros(kmax);
q(:,1)=unifrnd(0,kmax)*rand(2,1);
[v,D]=eig(A);
%%
% Power iteration
for i=1:kmax
    z(:,i+1)=A*q(:,i);
    q(:,i+1)= z(:,i+1)/norm(z(:,i+1),2);
    lambda(i)=q(:,i+1)'*A*q(:,i+1);
    fprintf('%d\n',lambda(i));
end
%%
figure(1)
plot(lambda);
%%
% Inverse iteration
miu=5;
for i=1:kmax
    z(:,i+1)=(A-miu*eye(2,2))^(-1)*q(:,i);
    q(:,i+1)= z(:,i+1)/norm(z(:,i+1),2);
    lambda(i)=q(:,i+1)'*A*q(:,i+1);
    fprintf('%d\n',lambda(i));
end
%%
figure(2)
plot(lambda);