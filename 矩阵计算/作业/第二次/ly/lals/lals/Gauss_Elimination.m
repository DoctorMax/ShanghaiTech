function U = Gauss_Elimination(B)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

[m, n] = size(B);
t = zeros(m, 1);
U = B;
for k = 1:n-1
    rows = k+1:m;
    cols = k+1:n;
    t(rows) = U(rows, k) / U(k, k);
    U(rows, cols) = U(rows, cols) - t(rows)*U(k, cols);
    U(rows, k) = 0;
%     L(rows, k) = t(rows);
end



end

