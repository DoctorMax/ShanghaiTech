clear;
clc;

% A = [2 2 4; 1 5 1; 1 1 8];
% % A_n = diag(A, 0);
% b = [1; 1; 1];

nn = 1000;
[A, n] = generate_mat(nn);
b = ones(n, 1);

tic
[A_inv, f_det] = my_inv(A);
% E = A*A_inv;
x = A_inv*b;
toc

% x
x_true = A\b;
error = max(abs(x-x_true))

