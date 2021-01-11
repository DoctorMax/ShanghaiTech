function z=soft(x,sigma)
z=[];
for i=1:length(x)
    z(i)=sign(x(i))*max(abs(x(i))-sigma,0);
end
z=z';