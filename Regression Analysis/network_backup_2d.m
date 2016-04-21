close all
clear all
%load('data1.mat');
% DD = {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'};
% %% ID
% ID_cell = data1(:,4);
% Week_cell = data1(:,1);
% Day_cell = data1(:,2);
% Size_cell = data1(:,6);
% Time_cell = data1(:,3);
% Name_cell = data1(:,5);
% Dura_cell = data1(:,7);
% 
% % cell 2 mat
% ID = cell2mat(ID_cell);
% Week = cell2mat(Week_cell);
% Size = cell2mat(Size_cell);
% Time = cell2mat(Time_cell);
% Name = cell2mat(Name_cell);
% Dura = cell2mat(Dura_cell);
% 
% %processing for Day_cell, converting it from words to number
% Day = zeros(size(Day_cell,1),1);
% for i = 1:size(Day_cell,1)
%     flag = 0;
%     for j = 1:size(DD,2)
%         if flag == 0
%             if strcmp(Day_cell{i},DD{j})
%                 Day(i) = j;
%                 flag = 1;
%             end
%         end
%     end
% end
% 
% 
% Data = cat(2, Week, Day, Time, ID, Name, Dura);
% 
% ID_uni = unique(ID);
% 
% for kkk = 1:length(ID_uni)
%     index_temp = find(Data(:,4)==ID_uni(kkk));
%     Data_flow = Data(index_temp,:);
%     Size_flow = Size(index_temp);
% end

% change name of the mat, i.e., d_flow1, d_flow2...... s_flow1,s_flow2
 load('d_flow0.mat');
 load('s_flow0.mat');
Data = Data_flow;
Size = Size_flow;
N = size(Data,1);
Indices = crossvalind('Kfold', N, 10);
index_all = 1:N;



Comb = unique(Data(:,1:3),'rows','stable');
% begin 10 fold cross-validation
for i = 1:10
    index_test = find(Indices==i);
    index_train = setdiff(index_all,index_test);
    X = Data(index_train,:);
    Y = Size(index_train);
    X_test = Data(index_test,:);
    Y_test = Size(index_test);
     C = MultiPolyRegress(X,Y,2);
   
    scatter(C.yhat,C.Residuals)
   xlabel('predicted values')
   ylabel('Residuals')
    
%    [B,BINT,R,RINT,STATS] = regress(Y,X);
%    result(i) = norm(Y_test - X_test*B,2);
    %B = polyfit(X,Y,2)
%   RMSE_test(i) = norm(Y_test - X_test*C.Coefficients(2:end)-C.Coefficients(1),2);
%   RMSE_train(i) = norm(Y-C.yhat,2)
%     coef(:,i) = C.Coefficients; 
 
  C = MultiPolyRegress(X,Y,9);
    D = MultiPolyRegress(X,Y,10);
    E = MultiPolyRegress(X,Y,7);
    F = MultiPolyRegress(X,Y,8);
%      G = MultiPolyRegress(X,Y,5);
%     H = MultiPolyRegress(X,Y,6);
%    [B,BINT,R,RINT,STATS] = regress(Y,X);
% B = C.Coefficients;  
%RMSE_test_linear = norm(Y_test - X_test*C.Coefficients(2:end)-C.Coefficients(1),2);
% RR1 = MultiPolyVal(X_test,C.Coefficients,Y_test,1);
% RMSE_test_linear = norm(Y_test - RR1.yhat,2);
% RMSE_train_linear = norm(Y-C.yhat,2)
% 
% RR2 = MultiPolyVal(X_test,D.Coefficients,Y_test,2);
% RMSE_test_2 = norm(Y_test - RR2.yhat,2);
% RMSE_train_2 = norm(Y-D.yhat,2)
% 
% 
% RR3 = MultiPolyVal(X_test,E.Coefficients,Y_test,3);
% RMSE_test_3 = norm(Y_test - RR3.yhat,2);
% RMSE_train_3 = norm(Y-E.yhat,2)
% 
% 
% RR4 = MultiPolyVal(X_test,F.Coefficients,Y_test,4);
% RMSE_test_4 = norm(Y_test - RR4.yhat,2);
% RMSE_train_4 = norm(Y-F.yhat,2)
  RR9 = MultiPolyVal(X_test,C.Coefficients,Y_test,9);
 RMSE_test_9 = norm(Y_test - RR9.yhat,2);
 RMSE_train_9 = norm(Y-C.yhat,2)
% 
% 
 RR10 = MultiPolyVal(X_test,D.Coefficients,Y_test,10);
 RMSE_test_10 = norm(Y_test - RR10.yhat,2);
 RMSE_train_10 = norm(Y-D.yhat,2)
 
  RR7 = MultiPolyVal(X_test,E.Coefficients,Y_test,7);
 RMSE_test_7 = norm(Y_test - RR7.yhat,2);
 RMSE_train_7 = norm(Y-E.yhat,2)
% 
% 
 RR8 = MultiPolyVal(X_test,F.Coefficients,Y_test,8);
 RMSE_test_8 = norm(Y_test - RR8.yhat,2);
 RMSE_train_8 = norm(Y-F.yhat,2)
    
    % polynominal
    % B = polyfit(X,Y,2)
    % result(i) = norm(Y_test - polyval(B,X_test),2)
    
%     for k = 1:size(Comb,1)
%         ii = intersect(intersect(find(X_test(:,1)==Comb(k,1)),find(X_test(:,2)==Comb(k,2))),find(X_test(:,3)==Comb(k,3)));
%         YY1(k) = sum(Y_test(ii));
%         YY2(k) = sum(X_test(ii,:)*B);
%     
%     end
%     

%     plot(fitted_values,R);
%     xlabel('predicted values')
%     hold on
%     scatter(YY1,YY2,'o')
%     xlabel('actual values')
%     ylabel('predicted values')
%     
%     axis([0 1 0 1])
%     hold on
%     plot(fitted_values,R);
%     pause
   
end

% RMSE_train=[sum(RMSE_train_linear)/10, sum(RMSE_train_2)/10 ,sum(RMSE_train_3)/10 , sum(RMSE_train_4)/10]
%   RMSE_test= [sum(RMSE_test_linear)/10, sum(RMSE_test_2)/10 ,sum(RMSE_test_3)/10 , sum(RMSE_test_4)/10]
  
  RMSE_train=[sum(RMSE_train_7)/10 , sum(RMSE_train_8)/10,sum(RMSE_train_9)/10 , sum(RMSE_train_10)/10]
  RMSE_test= [sum(RMSE_test_7)/10 , sum(RMSE_test_8)/10 ,sum(RMSE_test_9)/10 , sum(RMSE_test_10)/10]
% order=[1,2,3,4]

order=[1,2,3,4,5,6,7,8,9,10]
train=[0.1696,0.1345,0.0868,0.0595,0.0424,0.0394,0.0373,0.0357,0.0343,0.0365]
plot(order,train)






