function smallS=smallest_singular_value1(m)
A=triu(ones(m))-0.9*eye(m);
[~,S,~]=svd(A);
smallS=min(diag(S));
end