function [y,k,loss]=gauss_it(A,b,x,x0,N,eps)
n=size(x0);
y=zeros(n);
loss=[];
for k=1:N
    for i=1:n
        sigma1=0;
        sigma2=0;
        for j=1:i-1
            sigma1=sigma1+A(i,j)*y(j);
        end
        for j=i+1:n
            sigma2=sigma2+A(i,j)*x0(j);
        end
        y(i)=(b(i)-sigma1-sigma2)/A(i,i);
    end
    loss=[loss,norm(x-y,2)];
    if (norm(y-x0,2)<eps)
        break;
    end
    x0=y;
end
end