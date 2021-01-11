function x = Inv_solver(A, b)
% x = inv(A)*b

[A_inv, f_det] = my_inv(A);
x = A_inv*b;

end

