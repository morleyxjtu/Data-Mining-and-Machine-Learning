close all
clear all
load('data1.mat');
DD = {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'};
%% ID
ID_cell = data1(:,4);
Week_cell = data1(:,1);
Day_cell = data1(:,2);
Size_cell = data1(:,6);
Time_cell = data1(:,3);
Name_cell = data1(:,5);
Dura_cell = data1(:,7);

% cell 2 mat
ID = cell2mat(ID_cell);
Week = cell2mat(Week_cell);
Size = cell2mat(Size_cell);
Time = cell2mat(Time_cell);
Name = cell2mat(Name_cell);
Dura = cell2mat(Dura_cell);

%processing for Day_cell, converting it from words to number
Day = zeros(size(Day_cell,1),1);
for i = 1:size(Day_cell,1)
    flag = 0;
    for j = 1:size(DD,2)
        if flag == 0
            if strcmp(Day_cell{i},DD{j})
                Day(i) = j;
                flag = 1;
            end
        end
    end
end


N = size(Day,1);
Indices = crossvalind('Kfold', N, 10);
Data = cat(2, Week, Day, Time, ID, Name, Dura);
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
    C = MultiPolyRegress(X,Y,1);
    D = MultiPolyRegress(X,Y,2);
    E = MultiPolyRegress(X,Y,3);
    F = MultiPolyRegress(X,Y,4);
%    [B,BINT,R,RINT,STATS] = regress(Y,X);
% B = C.Coefficients;  
%RMSE_test_linear = norm(Y_test - X_test*C.Coefficients(2:end)-C.Coefficients(1),2);
RR1 = MultiPolyVal(X_test,C.Coefficients,Y_test,1);
RMSE_test_linear = norm(Y_test - RR1.yhat,2);
RMSE_train_linear = norm(Y-C.yhat,2)

RR2 = MultiPolyVal(X_test,D.Coefficients,Y_test,2);
RMSE_test_2 = norm(Y_test - RR2.yhat,2);
RMSE_train_2 = norm(Y-D.yhat,2)


RR3 = MultiPolyVal(X_test,E.Coefficients,Y_test,3);
RMSE_test_3 = norm(Y_test - RR3.yhat,2);
RMSE_train_3 = norm(Y-E.yhat,2)


RR4 = MultiPolyVal(X_test,F.Coefficients,Y_test,4);
RMSE_test_4 = norm(Y_test - RR4.yhat,2);
RMSE_train_4 = norm(Y-F.yhat,2)

%    coef(:,i) = B;
% scatter(C.yhat,C.Residuals)
% xlabel('predicted values')
% ylabel('Residuals')
%     i
  
    
%     for k = 1:size(Comb,1)
%         ii = intersect(intersect(find(X_test(:,1)==Comb(k,1)),find(X_test(:,2)==Comb(k,2))),find(X_test(:,3)==Comb(k,3)));
%         YY1(k) = sum(Y_test(ii));
%         YY2(k) = sum(X_test(ii,:)*B(2:end)+B(1));
%     
%     end

%     plot(fitted_values,R);
%     xlabel('fitted values')
%     hold on
%     scatter(YY1,YY2,'o')
%     legend('residual','actual values')
    %pause
end

 RMSE_train=[sum(RMSE_train_linear)/10, sum(RMSE_train_2)/10 ,sum(RMSE_train_3)/10 , sum(RMSE_train_4)/10]
 RMSE_test= [sum(RMSE_test_linear)/10, sum(RMSE_test_2)/10 ,sum(RMSE_test_3)/10 , sum(RMSE_test_4)/10]
order=[1,2,3,4]



