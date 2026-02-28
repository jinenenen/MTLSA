function[cindex]=getcindex_nocox(predict,times,status) 
%a function to return the Harrell's cindex(c-statistic) for the cox model
%函数定义：输入：predict：模型预测的风险值（MTLSA 中是sum(result,2)，即各时间任务预测值求和）；times：样本的实际观测时间（生存时间 / 删失时间）；status：删失指示（1 = 未删失，0 = 删失）；输出：cindex：计算得到的一致性指数。

%初始化 4 个累加变量（核心是区分 “一致的样本对” 和 “可比较的样本对”）
sum1=0;
sum2=0;
sum3=0;
sum4=0;

for i=1:1:(size(predict,1)-1) %双重循环：遍历所有无序样本对（i<j，避免重复计算 i-j 和 j-i），对每一对样本 (i,j) 进行判断。
    for j=i+1:1:size(predict,1)
        stime1=times(i); % 样本i的观测时间
        stime2=times(j); % 样本j的观测时间
        pred1=predict(i); % 样本i的预测风险值  不对，应该是生存概率
        pred2=predict(j); % 样本j的预测风险值
        status1=status(i); % 样本i的删失状态
        status2=status(j); % 样本j的删失状态
        
        
        if stime1<stime2 && pred1<pred2 && status1==1 % 场景1：样本i未删失，且i的观测时间 < j的观测时间
            sum1=sum1+1; % 预测一致（i预测 < j预测） 若stime1 < stime2（i 死得更早），则pred1（i的生存概率） < pred2（j的生存概率） → 预测一致，计入sum1。
        end
        if stime2<stime1 && pred2<pred1 && status2==1 % 场景2：样本j未删失，且j的观测时间 < i的观测时间
             sum2=sum2+1; % 预测一致（j预测 < i预测）
        end
        if stime1<stime2 && status1==1 % 累计场景1的可比较样本对总数
            sum3=sum3+1;
        end
        if stime2<stime1 && status2==1 % 累计场景2的可比较样本对总数
            sum4=sum4+1;
        end
    end
end
cindex=(sum1+sum2)/(sum3+sum4);  %两者之和sum3+sum4 = 所有可比较样本对的总数。
end
