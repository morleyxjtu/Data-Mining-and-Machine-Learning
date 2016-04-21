close all
clear all
load('data2.mat');

N = size(data2,1);
index_all = 1:N;
Indices = crossvalind('Kfold', N, 10);

alpha = 0.1;

for i = 1:10
    index_test = find(Indices==i);
    index_train = setdiff(index_all,index_test);
    X = data2(index_train,1:13);
    Y = data2(index_train,14);
    X_test = data2(index_test,1:13);
    Y_test = data2(index_test,14);
   
    n=length(Y)
    cvx_begin
    variable B(13)
    minimize( Y'*Y - Y'*X*B -B'*X'*Y + B'*X'*X*B + alpha*B'*B)
   % minimize( (Y'*Y - Y'*X*B -B'*X'*Y + B'*X'*X*B)/(2*n) + alpha*norm(B,1))
cvx_end
  
  %  B = fminsearch(norm(Y-X*B,2)^2+alpha*norm(B,2)^2,0); 
   RMSE_train=norm(Y-X*B,2)
   RMSE_test = norm(Y_test - X_test*B,2);
    coef(:,i) = B;
    i
    pause
end



