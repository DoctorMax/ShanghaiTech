clc;
clear;
%% 设置参数
mMax=30;
smallS1=zeros(mMax,1);
smallS2=zeros(mMax,1);
%% 调用函数
for i=1:mMax
    smallS1(i)=smallest_singular_value1(i);
    smallS2(i)=smallest_singular_value2(i);
end
%% 打印输出
figure(1)
grid on
loglog(smallS1);
hold on
loglog(abs(smallS2));