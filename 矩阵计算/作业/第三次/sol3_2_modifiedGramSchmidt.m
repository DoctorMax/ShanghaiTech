clear;
clc;
epsilon=1e-4;
% epsilon=1e-9;
A=[1,1,1;epsilon,epsilon,0;epsilon,0,epsilon];
[m,n] = size(A);
R1=zeros(n,n);
Q1=zeros(m,n);
%正交�?
for i=1:n
    z=A(:,i);
    R1(i,i)=norm(z,2);
    Q1(:,i)=z/R1(i,i);
    R1(i,i+1:n)=Q1(:,i)'*A(:,i+1:n);
    A(:,i+1:n)=A(:,i+1:n)-Q1(:,i)*R1(i,i+1:n);
end
Q1

norm(Q1'*Q1-eye(n),'fro')