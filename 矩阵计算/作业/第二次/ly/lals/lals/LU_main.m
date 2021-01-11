clear;
clc;

% A = [2 2 5; 1 9 1; 3 1 7];
% n = size(A, 1);
% b = [1; 1; 1];

nn = 1000;
[A, n] = generate_mat(nn);
b = ones(n, 1);


z = zeros(n, 1);
x = zeros(n, 1);

tic
%% LU decomposition
[L, U] = my_lu(A);

% [L_true, U_truem P] = lu(A);


%% forward substitution
for i = 1:n
    tmp = b(i);
    for j = 1:i-1
        tmp = tmp - z(j)*L(i, j);
    end
    z(i) = tmp / L(i, i);
end


%% backward substitution
for i = n:-1:1
    tmp = z(i);
    for j = n:-1:i+1
        tmp = tmp - x(j)*U(i, j);
    end
    x(i) = tmp / U(i, i);
end
toc

t = toc

x;
x_true = A\b;
error = max(abs(x-x_true))



