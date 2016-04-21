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

% cell 2 mat
ID = cell2mat(ID_cell);
Week = cell2mat(Week_cell);
Size = cell2mat(Size_cell);
Time = cell2mat(Time_cell);

%processing for Day_cell
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

% combine week and day toghther
Day_abs = (Week-1)*7 + Day;


% only need 20 days, so created a newly satisfied data
Index = find(Day_abs<21);
ID_new = ID(Index);
Day_abs_new = Day_abs(Index);
Size_new = Size(Index);
Time_new = Time(Index);

ID_uni = unique(ID_new);
for i = 1:length(ID_uni)
    index1 = find(ID_new==ID_uni(i)); % the indexes of the IDs that belong to the same workflow
    
    ID_temp = ID_new(index1);
    Day_abs_temp = Day_abs_new(index1);
    Size_temp = Size_new(index1);
    Time_temp = Time(index1);
    
    Comb = unique(cat(2,Day_abs_temp,Time_temp),'rows','stable');
    for j=1:size(Comb,1)
        index2 = intersect(find(Day_abs_temp==Comb(j,1)),find(Time_temp==Comb(j,2)));
        s(j) = sum(Size_temp(index2));
    end
    plot(s)
    xlabel('change the name and scale by yourself')
    ylabel('change the name and scale by yourself')
    fprintf('please save the figure, then press any key to move further');
    pause
    
    
    
    
    %for j = 1:(size(PP,1)-19)
    %    index2 = intersect(find(Day_abs>=PP(j)), find(Day_abs<=PP(j)+19)); % the indexes of the IDs that belong to the qualified period
    %    index_final = intersect(index,index2);
    %    file_size(i,j) = sum(Size(index_final));
    %end
    
    % PP = [0,20,40,60,80,100];
    %     for j = 1:(size(PP,2)-1)
    %         index2 = intersect(find(Day_abs>PP(j)), find(Day_abs<=PP(j+1))); % the indexes of the IDs that belong to the qualified period
    %         index_final = intersect(index,index2);
    %         file_size(i,j) = sum(Size(index_final));
    %     end
end

