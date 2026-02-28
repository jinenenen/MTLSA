%% file MTLSA.m
% this file shows the process of ADMM methods of the MTLSA model
% "Multi-Task Learning model for Survival Analysis"
%to learn a parth wise solution of MTLSA
%
%% OBJECTIVE
% argmin_XB \in P 0.5 * norm (Wo(Y - X * B))^2 
%            + \lambad_1 * \|B\|_{2,1} + \lambad_2 * \|B\|_F^2}
%
%% RELATED PAPERS
%  [1]Yan Li, Jie Wang, Jieping Ye and Chandan K. Reddy "A Multi-Task Learning
%     Formulation for Survival Analysis". In Proceedings of the 22nd ACM SIGKDD
%     International Conference on Knowledge Discovery and Data Mining (KDD'16),
%     San Francisco, CA, Aug. 2016
%
%% RELATED PACKAGES 
%  SLEP, MALSAR
%% INPUT
% floder: - the direction where it contains train/test data
% name_train:  - name of training data (.mat is not needed)
% name_test: - name of testing data (.mat is not needed)
% lam_iter: - number of searched lambdas 
% Smallest_lambda_rate: - smallest_lambda/lambda_max, usually set as 0.01
%% Run Example 
%  MTLSA 'NSBCD_data/' 'NSBCD_train_1' 'NSBCD_test_1' 100 0.01

function MTLSA(floder, name_train, name_test,lam_iter,Smallest_lambda_rate)
current_path=cd;
Num_lambda=str2num(lam_iter); % 正则化系数搜索次数（如100）
smallest_rate=str2double(Smallest_lambda_rate); % 最小λ₁/最大λ₁的比例（如0.01）
addpath(genpath([current_path '/functions/'])); % load function % 添加SLEP/MALSAR工具包路径

% tell the direction where it contains train/test data. % 读取数据（由survival_data_pre.m生成的.mat文件）
dir=strcat(current_path,'/data/',floder); 
load(strcat(dir,name_train,'.mat')); % load training data. % 加载训练集X/W/Y/Time/Status
load(strcat(dir,name_test,'.mat')); % load testing data. % 加载测试集X_test/W_test/Y_test/Time_test/Status_test
d = size(X, 2);  % dimensionality. % 特征维度
max_day = size(Y,2); % 时间任务数（max_time）

% 统一测试集Y_test的维度（和训练集max_day一致）
Y_test=Y_test(:,1:max_day);
Y=Y(:,1:max_day);
W=W(:,1:max_day);

% 优化器参数（ADMM的终止条件）
opts.init = 0;      % guess start point from data.  % 0=从数据初始化，1=温启动（后续迭代复用前一次参数）
opts.tFlag = 1;     % terminate after relative objective value does not changes much.  % 按目标函数相对变化终止
opts.tol = 10^-4;   % tolerance.  % 收敛容忍度
opts.maxIter = 1000; % maximum iteration number of optimization. % 优化最大迭代次数

%%build the output matrix  % 存储不同λ₁下的结果
sparsity = zeros(Num_lambda, 1);
cindex=zeros(Num_lambda, 1);

num_sample = size(Y_test,1);
num_task = size(Y_test,2);

AUC_matrix=zeros(Num_lambda,num_task);
contains=zeros(num_task,1);
dimension = size(X, 2);

%% TRAIN
%%Initialize the parameter 
rho=10;
max_iter=100;
B_old = zeros(dimension, num_task);
mu = zeros(size(Y,1),num_task);
Mold=Y;

%%Calculate the smallest possible \lambad_1 which will make B=0
M=update_M(X, Y,Mold, rho, B_old, W, mu,opts);
newfirstY=M+mu;
max_lambda = get_lambda_max(X, newfirstY,rho);

%%pawise wise search for best \lambad_1
lambda = zeros(1,Num_lambda);
for i=1:Num_lambda
    lambda(i)=max_lambda*(smallest_rate)^(i/Num_lambda);
end
log_lam  = log(lambda);

ALL_B=cell(1,Num_lambda);
tic;    
for i = 1: Num_lambda
    dif=1;
    iter=1;
    disp(i)
    %%ADMM for MTLSA
    while (dif>(5*10^-4) && iter<=max_iter)

        %Step 1: update M
        M=update_M(X, Y,Mold, rho, B_old, W, mu,opts);
        %Step 2: update B standard L_{2,1} solver
        newY=M+mu;
        Mold=M;
        [B funcVal_B] = Least_L21_Standard(X, newY, lambda(i),rho,B_old, opts);
        %Update mu
        mu=mu+M-X*B;
        dif=norm ((B-B_old),'fro');
        B_old=B;
        iter=iter+1;
    end
    % set the solution as the next initial point to get Warm-start. 
    opts.init = 1;
    opts.W0 = B;

    sparsity(i) = nnz(sum(B,2 )==0)/d;
    ALL_B{i}=B; %contains all B's with respect to different lambda
end
toc; %output the training time

%% TESTING %% 测试阶段：对每个λ₁下的模型评估性能
for i = 1: Num_lambda
    result=X_test*ALL_B{i}; % 1. 预测：测试集特征 × 模型参数B → 各时间任务的预测值
     %call the sequence_bottomup function to make sure the prediction  % 2. 修正预测值：保证非负且非递增
     %follows the non-negative non-increasing list structure   % sequence_bottomup：文献中提出的单调性约束函数
    for ii = 1:num_sample
        result(ii,:)=sequence_bottomup(result(ii,:),num_task);
    end
    % evaluate the model performance by concordance index % 3. 计算C-index（生存时间排序一致性）
    cindex(i)=getcindex_nocox(sum(result,2),Time_test,Status_test); % sum(result,2)：将各时间任务的预测值求和，作为整体生存风险值
    % evaluate the model performance by calculating AUC for each task % 4. 计算每个时间任务的AUC
    for k =1:num_task
        temp=find(W_test(:,k)); %找到第k个任务的有效样本（W_test=1）
        label=Y_test(temp,k); % 有效样本的真实标签（0/1/0.5，0.5会被过滤）
        contains(k)=size(temp,1); % 第k个任务的有效样本数
        if length(unique(label))>1 % 标签需包含0和1，否则无法计算AUC
            pred=result(temp,k); % 有效样本的预测值
            [X_pred,Y_Pred,T_Pred,AUC_Pred] = perfcurve(label,pred,1); % perfcurve：MATLAB内置函数，计算二分类AUC
            AUC_matrix(i,k)=AUC_Pred;
        end
    end
    
end

%calculating the weighted average of AUC % 5. 计算加权平均AUC（按各任务有效样本数加权）
haveAUC=find(AUC_matrix(1,:)); % 找到有有效AUC的任务
weighted_AUC=(AUC_matrix(:,haveAUC)*contains(haveAUC,:))/sum(contains(haveAUC));

% 输出最优结果
X_disp = ['Best possible weighted AUC is: ',num2str(max(weighted_AUC)),...
    ' and the Best possible Cindex is: ',num2str(max(cindex))];
disp(X_disp)
disp(['Please check the "',name_test,Smallest_lambda_rate,'_result.mat" file to check all the results']) 
disp('with respect to different lambdas and select the best lambda for your own dataset.')

% draw figure % 绘制λ₁与稀疏度的关系图
h = figure;
plot(log_lam, sparsity);
xlabel('log(\lambda_1)')
ylabel('Row Sparsity of Model (Percentage of All-Zero Columns)')
title('Row Sparsity of Predictive Model when Changing Regularization Parameter');

% 保存所有结果
save(strcat(dir,name_test,Smallest_lambda_rate,'_result.mat'),'ALL_B',...
    'weighted_AUC','cindex','lambda','AUC_matrix','sparsity');
end
