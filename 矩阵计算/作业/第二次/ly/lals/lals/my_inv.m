function [A_inv, f_det] = my_inv(A)
% Computte the inverse of nonsingular square matrix A
%   Gauss-Jordan method

n = size(A, 1);

i_idx = zeros(n, 1);
j_idx = zeros(n, 1);
f = 1;
f_det = 1.0;

for k = 1:n
    % step1: find the pivots
    f_max = 0.0;
    for i = 1:n
        for j = 1:n
            ff = abs(A(i, j));
            if ff > f_max
                f_max = ff;
                i_idx(k) = i;
                j_idx(k) = j;
            end
        end
    end
%     f_max
    if abs(f_max) < 1e-3
        disp("A maybe sigular...")
        return;
    end
    
    if i_idx(k) ~= k
        f = -f;
        swap_mat(A, k, i_idx(k), 1);
    end
    if j_idx(k) ~= k
        f = -f;
        swap_mat(A, k, j_idx(k), 2);
    end
    
    % compute the det
    f_det = f_det*A(k, k);
    
    % step2: compute the inverse of A
    A(k, k) = 1.0 / A(k, k);
    
    % step3
    for j = 1:n
        if j ~= k
            A(k, j) = A(k, j)*A(k, k);
        end
    end
    
    % step4
    for i = 1:n
        if i ~= k
            for j = 1:n
                if j ~= k
                    A(i, j) = A(i, j) - A(i, k)*A(k, j);
                end
            end
        end
    end
    
    % step5
    for i = 1:n
        if i ~= k
            A(i, k) = -A(k, k) * A(i, k);
        end
    end
end

for k = n:-1:1
    if i_idx(k) ~= k
        swap_mat(A, k, i_idx(k), 2);
    end
    if j_idx(k) ~= k
        swap_mat(A, k, j_idx(k), 1);
    end
end

A_inv = A;

f_det = f_det*f;

end


function A_new = swap_mat(A, k1, k2, axis)
% Exchange rows(cols) of matrix A
%       axis == 1, exchange rows
%       axis == 2, exchange cols

if axis == 1
    tmp = A(k1, :);
    A(k1, :) = A(k2, :);
    A(k2, :) = tmp;
elseif axis == 2
    tmp = A(:, k1);
    A(:, k1) = A(:, k2);
    A(:, k2) = tmp;
else
    throw("The value of axis is invalid...");
end

A_new = A;

end



