function smallS=smallest_singular_value2(m)
A=triu(ones(m))-0.9*eye(m);
S=zeros(size(A));
[~,E]=eig(A'*A);
for i=1:size(A,2)
    S(i,i)=sqrt(E(i,i));
end
smallS=min(diag(S));
end