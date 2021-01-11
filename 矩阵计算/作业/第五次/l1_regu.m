function f=l1_regu(A,x,y,lambda)
f=norm(A*x-y,2)^2+lambda*norm(x,1);
end