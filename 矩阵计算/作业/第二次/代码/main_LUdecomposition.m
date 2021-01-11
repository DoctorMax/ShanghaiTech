clear;
clc;

A = [2 2 5; 1 9 1; 3 1 7];
n = size(A, 1);
b = [1; 1; 1];
z = zeros(n, 1);
x = zeros(n, 1);

tic
% LU decomposition
[L, U] = lu_my(A);
% forward substitution
for i = 1:n
    tmp1 = b(i);
    for j = 1:i-1
        tmp1 = tmp1 - z(j)*L(i, j);
    end
    z(i) = tmp1 / L(i, i);
end
% backward substitution
for i = n:-1:1
    tmp2 = z(i);
    for j = n:-1:i+1
        tmp2 = tmp2 - x(j)*U(i, j);
    end
    x(i) = tmp2 / U(i, i);
end
toc
t = toc;
fprintf('t=%d\n',t);
x_true = A\b;
error = max(abs(x-x_true));
fprintf('error=%d\n',error);

function [L, U] = lu_my(A)
    %LU decomposition
    n = size(A, 1);
    L = eye(n);
    U = A;
    for i = 1:n-1
        j = i+1:n;
        L(j, i) = U(j, i) / U(i, i);
        U(j, j) = U(j, j) - L(j, i)*U(i, j);
        U(j, i) = 0;
    end
end


