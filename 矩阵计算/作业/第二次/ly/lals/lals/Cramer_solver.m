function x = Cramer_solver(A, b)
% Cramer rule

n = size(A, 1);
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

end

