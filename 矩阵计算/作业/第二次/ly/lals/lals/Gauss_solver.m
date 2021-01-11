function x = Gauss_solver(A, b)
% Gauss elimination

B = [A b];
[m, n] = size(B);
x = zeros(m, 1);

U = Gauss_Elimination(B);

for i = m:-1:1
    tmp = U(i, n);
    for j = n-1:-1:i+1
        tmp = tmp - U(i, j)*x(j);
    end
    x(i) = tmp / U(i, i);
end


end

