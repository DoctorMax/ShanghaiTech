function [y,k,loss]=jacobi_it(A,b,x,x0,N,eps)
n=size(x0);
y=zeros(n);
loss=[];
for k=1:N
    for i=1:n
        sigma=0;
        for j=1:n
            if i~=j
                sigma=sigma+A(i,j)*x0(j);
            end
        end
        y(i)=(b(i)-sigma)/A(i,i);
    end
    loss=[loss,norm(x-y,2)];
    if (norm(y-x0,2)<eps)
        break;
    end
    x0=y;
end
end