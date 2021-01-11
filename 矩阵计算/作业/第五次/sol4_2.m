clc;
close all;
clear;
%% 读取数据
load('./data_problem4/all_data.mat');
%% 设置参数
X=data_train;
X_test=data_test;
num=size(X,2);
d=10;
%% 算法实现
miu=mean(X,2);
[U,S,V]=svd(X-miu);
[S_sort,index]=sort(diag(S),'DESCEND');
S_sort_ex=diag(S_sort);
S_sort_ex(num:size(X,1),:)=0;
U_sort=U(:,index);
U_d=U_sort(:,1:d);
miuface=reshape(miu,48,42);
for i=1:d
    temp=U_d(:,i);
    U_d_norml(:,i)=(temp-min(temp))/(max(temp)-min(temp));
end
eigenface=reshape(U_d_norml,48,42,10);
Y_test=U_d'*(X_test-miu);
X_test_proj=reshape(miu+U_d*Y_test,48,42,20);
%% 打印输出
figure(1)
imshow(miuface);
figure(2)
hold on
subplot 221;
imshow(eigenface(:,:,1));
title('the first eigenface');
subplot 222;
imshow(eigenface(:,:,2));
title('the second eigenface');
subplot 223;
imshow(eigenface(:,:,3));
title('the third eigenface');
subplot 224;
imshow(eigenface(:,:,10));
title('the tenth eigenface');
figure(3)
plot(S_sort);
figure(4)
hold on
subplot 221;
imshow(X_test_proj(:,:,1));
title('the first projected face');
subplot 222;
imshow(X_test_proj(:,:,6));
title('the second projected face');
subplot 223;
imshow(X_test_proj(:,:,11));
title('the third projected face');
subplot 224;
imshow(X_test_proj(:,:,16));
title('the fourth projected face');