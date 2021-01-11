function [L, U] = my_lu(A)
% The LU decomposition of A

n = size(A, 1);
L = eye(n);
U = A;


%% lu decomposition
for k = 1:n-1
    rows = k+1:n;
    L(rows, k) = U(rows, k) / U(k, k);
    U(rows, rows) = U(rows, rows) - L(rows, k)*U(k, rows);
    U(rows, k) = 0;
end


end

