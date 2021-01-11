function [x] = LU_solver(A, b)
% LU decomposition

n = size(A,  1);
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



end

