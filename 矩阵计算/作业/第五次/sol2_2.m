clc;
clear;
%% 参数设置
B=[3,2;1,4;0.5,0.5];
% B=randn(3,2);
S1=zeros(size(B));
S2=zeros(size(B));
%% 实现算法
[V1,E1]=eig(B'*B);
for i=1:size(B,2)
    S1(i,i)=sqrt(E1(i,i));
end
[Q1,~]=qr(B*V1);
U1=Q1;
loss1=B-U1*S1*V1';
%% 优化算法
[V2,E2]=eig(B'*B);
for i=1:size(B,2)
    S2(i,i)=sqrt(E2(i,i));
end
[S2_sort,index]=sort(diag(S2),'DESCEND');
S2 = [diag(S2_sort);0,0];
V3=V2(:,index);
[Q2,R2]=qr(B*V3);
U2=Q2;
V4=[-V3(:,1),-V3(:,2)];
loss2=B-U2*S2*V4';
[UU,SS,VV]=svd(B);
%% 打印输出
disp(B);
disp(U2*S2*V4');