clear;
clc;

% A = [2 2 4; 1 5 1; 1 1 8];
% n = size(A, 1);
% b = [1; 1; 1];

nn = 1000;
[A, n] = generate_mat(nn);
b = ones(n, 1);


A_det = det(A);
x = zeros(n, 1);
tic
for j= 1:n
    tmp = A(:, j);
    A(:, j) = b;
    det1 = det(A);
    x(j) = det1/A_det;
    A(:, j) = tmp;
end
toc

% x
x_true = A\b;
error = max(abs(x-x_true))
