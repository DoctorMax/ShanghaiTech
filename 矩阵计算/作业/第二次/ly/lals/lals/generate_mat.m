function [A, n2] = generate_mat(nn)
% Generate the coefficient matrix A of the linear system
%   triangular matrix

n = round(sqrt(nn));
n2 = n*n;

%% Consturct the coefficient matrix
r = 1.2*ones(n, 1);
r1 = -ones(n-1, 1);
B = diag(r, 0) + diag(r1, -1) + diag(r1, 1);

I = eye(n);

A = kron(I, B) + kron(B, I);

% b = ones(n*n, 1);


end

