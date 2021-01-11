clear;
clc;
epsilon=1e-4;
% epsilon=1e-9;
A=[1,1,1;epsilon,epsilon,0;epsilon,0,epsilon];
[m,n] = size(A);
Q=zeros(m,n);
%正交�?
Q(:,1)=A(:,1);
for i=2:n
    for j=1:i-1
        Q(:,i)=Q(:,i)-dot(A(:,i),Q(:,j))/dot(Q(:,j),Q(:,j))*Q(:,j);
    end
    Q(:,i)=Q(:,i)+A(:,i);
end

%单位�?
 for k=1:n
     Q(:,k)=Q(:,k)/norm(Q(:,k));
 end

Q
 
norm(Q'*Q-eye(n),'fro')