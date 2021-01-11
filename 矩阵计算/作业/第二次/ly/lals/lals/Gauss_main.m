clear;
clc;

% A = [2 2 4; 1 5 1; 1 1 8];
% % n = size(A, 1);
% b = [1; 1; 1];

nn = 1000;
[A, n] = generate_mat(nn);
b = ones(n, 1);


B = [A b];
[m, n] = size(B);
x = zeros(m, 1);

tic
U = Gauss_Elimination(B);

% U

for i = m:-1:1
    tmp = U(i, n);
    for j = n-1:-1:i+1
        tmp = tmp - U(i, j)*x(j);
    end
    x(i) = tmp / U(i, i);
end
toc

% x

x_true = A\b;
error = max(abs(x-x_true))


% U_true = rref(B)
% rrefm
% rrefmovie(B)

