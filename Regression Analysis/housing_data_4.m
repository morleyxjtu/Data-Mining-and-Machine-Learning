close all
clear all
load('data2.mat');

N = size(data2,1);
index_all = 1:N;
Indices = crossvalind('Kfold', N, 10);

for i = 1:10
    index_test = find(Indices==i);
    index_train = setdiff(index_all,index_test);
    X = data2(index_train,1:13);
    Y = data2(index_train,14);
    X_test = data2(index_test,1:13);
    Y_test = data2(index_test,14);
   
%     B = MultiPolyRegress(X,Y,1,'figure');
   
%     C = MultiPolyRegress(X,Y,2,'figure');
%   D = MultiPolyRegress(X,Y,3,'figure');
     E = MultiPolyRegress(X,Y,4,'figure');
%     F = MultiPolyRegress(X,Y,5);
%     G = MultiPolyRegress(X,Y,6);
%     H = MultiPolyRegress(X,Y,7);
%     I = MultiPolyRegress(X,Y,8);

RR1 = MultiPolyVal(X_test,B.Coefficients,Y_test,1);
RMSE_test_linear = norm(Y_test - RR1.yhat,2);
RMSE_train_linear = norm(Y-B.yhat,2)

RR2 = MultiPolyVal(X_test,C.Coefficients,Y_test,2);
RMSE_test_2 = norm(Y_test - RR2.yhat,2);
RMSE_train_2 = norm(Y-C.yhat,2)


RR3 = MultiPolyVal(X_test,D.Coefficients,Y_test,3);
RMSE_test_3 = norm(Y_test - RR3.yhat,2);
RMSE_train_3 = norm(Y-D.yhat,2)


RR4 = MultiPolyVal(X_test,E.Coefficients,Y_test,4);
RMSE_test_4 = norm(Y_test - RR4.yhat,2);
RMSE_train_4 = norm(Y-E.yhat,2)
% 
%   RR5 = MultiPolyVal(X_test,F.Coefficients,Y_test,5);
%  RMSE_test_5 = norm(Y_test - RR5.yhat,2);
%  RMSE_train_5 = norm(Y-F.yhat,2)
% 
% 
%  RR6 = MultiPolyVal(X_test,G.Coefficients,Y_test,6);
%  RMSE_test_6 = norm(Y_test - RR6.yhat,2);
%  RMSE_train_6 = norm(Y-G.yhat,2)
%  
%   RR7 = MultiPolyVal(X_test,H.Coefficients,Y_test,7);
%  RMSE_test_7 = norm(Y_test - RR7.yhat,2);
%  RMSE_train_7 = norm(Y-H.yhat,2)
% 
% 
%  RR8 = MultiPolyVal(X_test,I.Coefficients,Y_test,8);
%  RMSE_test_8 = norm(Y_test - RR8.yhat,2);
%  RMSE_train_8 = norm(Y-I.yhat,2)
%  
   scatter(B.yhat,B.Residuals)
   xlabel('predicted values')
   ylabel('Residuals')
   
   scatter(E.yhat,E.Residuals)
   xlabel('predicted values')
   ylabel('Residuals')
    % polynominal
    % B = polyfit(X,Y,2)
    % result(i) = norm(Y_test - polyval(B,X_test),2)
  
end

% RMSE_train=[sum(RMSE_train_linear)/10, sum(RMSE_train_2)/10 ,sum(RMSE_train_3)/10 , sum(RMSE_train_4)/10,sum(RMSE_train_5)/10, sum(RMSE_train_6)/10 ,sum(RMSE_train_7)/10 , sum(RMSE_train_8)/10]
%  RMSE_test= [sum(RMSE_test_linear)/10, sum(RMSE_test_2)/10 ,sum(RMSE_test_3)/10 , sum(RMSE_test_4)/10 ,sum(RMSE_test_5)/10, sum(RMSE_test_6)/10 ,sum(RMSE_test_7)/10 , sum(RMSE_test_8)/10]
order=[1,2,3,4,5,6,7,8]

RMSE_train=[sum(RMSE_train_linear)/10, sum(RMSE_train_2)/10 ,sum(RMSE_train_3)/10 , sum(RMSE_train_4)/10]
RMSE_test= [sum(RMSE_test_linear)/10, sum(RMSE_test_2)/10 ,sum(RMSE_test_3)/10 , sum(RMSE_test_4)/10]

