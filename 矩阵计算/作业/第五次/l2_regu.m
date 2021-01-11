function f=l2_regu(A,x,y,lambda)
f=norm(A*x-y,2)^2+lambda*norm(x,2)^2;
end