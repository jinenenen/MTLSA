%The ?survival_data_pre.m? is used to generate the target matrix Y, feature matrix X, 
%and the indicator matrix W, from the original training and testing data.
%The original training and testing files are both in ".csv" format.
%Where each instance is represented as a row in file and 
%the last two columns are survival_times and censored_indicators, respectively. 

function survival_data_pre (floder, name_train, name_test)
%输入 3 个参数，分别是floder（数据文件夹名）、name_train（训练集前缀）、name_test（测试集前缀）

%路径拼接
current_path=cd; %cd获取当前 MATLAB 工作路径
dir=strcat(cd,'/data/',floder); %dir拼接出数据文件夹的完整路径（如当前路径/data/NSBCD_data/）
train=strcat(dir,name_train,'.csv');
test=strcat(dir,name_test,'.csv');
data = csvread(train); %读取数据：csvread读取 CSV 文件为矩阵，每行是 1 个样本，最后两列是survival_times（生存时间）和censored_indicators（删失指示：0 = 删失，1 = 未删失）
data_test = csvread(test);

% the time intervial can be adjusted here.
data(:,end-1)=fix(data(:,end-1)); %fix()：向下取整，将生存时间转为整数（如 10.8 天→10 天），确保时间间隔是整数
data_test(:,end-1)=fix(data_test(:,end-1));
%% for example if the orginal survival time is dayly based, then you can
% devided by 7 to get weekly based time intervial.
%data(:,end-1)=fix(data(:,end-1)/7);
%data_test(:,end-1)=fix(data_test(:,end-1)/7);
%%

max_time=max(max(data(:,end-1)),max(data_test(:,end-1))); %确定最大时间间隔数：取训练集 + 测试集所有样本的生存时间最大值，作为max_time（即多任务的任务数，比如最大生存时间是 188 天，就有 188 个时间间隔任务）；
max_time
%拆分训练集数据
label=data(:,end-1:end); %label：提取最后两列（生存时间 + 删失指示）
X=data(:,1:end-2); % X：提取前end-2列（样本的原始特征），即特征矩阵
nsample=size(label,1); %nsample：训练集样本数
%初始化矩阵
W=ones(nsample,max_time); %W：初始化指示矩阵为全 1（默认所有时间间隔的标签都有效）
Y=zeros(nsample,max_time); %Y：初始化目标矩阵为全 0（默认所有时间间隔都标记为 “存活”）
for i=1:nsample;
    if label(i,2)==0  % 样本i是删失样本（censored_indicators=0）
        W(i,(label(i,1)+1):end)=0; % 删失时间之后的时间间隔，标签无效（W置0）
        Y(i,(label(i,1)+1):end)=0.5; % 删失时间之后的生存状态未知，Y置0.5（模型不使用这些位置的标签）
    end
    Y(i,1:label(i,1))=1; % 从第1个时间间隔到样本i的生存时间，Y置1（标记为“死亡”）
end
Time=label(:,1);
Status=label(:,2);
save(strcat(dir, name_train,'.mat'),'X','W','Y','Time','Status');

% 以下是测试集的相同逻辑，仅变量名加了_test，不再逐行解析
label_test=data_test(:,end-1:end);
X_test=data_test(:,1:end-2);
nsample_test=size(label_test,1);
W_test=ones(nsample_test,max_time);
Y_test=zeros(nsample_test,max_time);
for i=1:nsample_test;
    if label_test(i,2)==0
        W_test(i,(label_test(i,1)+1):end)=0;
        Y_test(i,(label_test(i,1)+1):end)=0.5;
    end
    Y_test(i,1:label_test(i,1))=1;
end
Time_test=label_test(:,1);
Status_test=label_test(:,2);
save(strcat(dir,name_test,'.mat'),'X_test','W_test','Y_test','Time_test','Status_test');
end
