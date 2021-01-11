clear;
clc;
data=importdata('data.txt');
label=importdata('label.txt');
x=rand (210, 1);
gamma=1e-5;
batch=0;
delta=2*(data'*data*x-data'*label);
while(max(abs(delta))>1e-8)
    delta=2*(data'*data*x-data'*label);
    x=x-gamma*delta;
    batch=batch+1;
end
gradientnorm=norm(delta);
loss=norm(label-data*x)^2;
save('sol1.txt','x','-ascii');