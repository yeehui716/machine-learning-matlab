function [H] = TrAdaBoost(TrainS,TrainA,LabelS,LabelA,Test,N,Learner)
%
% H 测试样本的最终分类标签
% TrainS 原训练数据
% TrainA 辅助训练数据
% LabelS 原训练数据标签(列向量)
% LabelA 辅助训练数据标签(列向量)
% Test  测试数据
% N 迭代次数
% Learner 基本分类器
% Write by ChenBo 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData  = [TrainS;TrainA]
trainLabel = [LabelS;labelA]

% rowS 原训练样本的个数, columnS 原训练样本的维数 ;  rowA 辅助训练样本的个数, columnA 辅助训练样本的维数
[rowS,columnS] = size(TrainS)
[rowA,columnA] = size(TrainA)
%rowT 测试样本的个数 , columnT测试样本的维数
[rowT,columnT] = size(test)

%算法需要计算所有训练、测试数据的预测结果
testData = [trainData;test]

%初始化训练样本权重及Beta
weight = ones(rowS+rowA,1)/(rowS+rowA)
beta  = 1/(1+sqrt(2*log(rowS/N)))
betaT = zeros(N,1)

%记录N次迭代训练的结果（包括训练数据与测试数据）
resultLabels = ones(N,size(rowS+rowA+rowT,1))

for i=1:N
    p = weight./sum(weight)
    resultLabels(:,i)= WeightKNN(TrainData,TrainLabel,testData,5, weight);
    er = ErrorRate(LabelS,resultLabels(1:rowS,i),weight)  %原训练数据的分类错误率
    if(er>0.5)
        er = 0.5
    end
    betaT(i)=er/(1-er)
    for j=1:rowS %更新原训练数据的权重
        weight(j) = weight(j)*bate^abs(result(j,i)-LabelS(j))
    end
    for j=1:rowA %更新辅助训练数据的权重
        weight(rowS+j) = weight(rowS+j)*btatT^(-abs(result(rowS+j,i))-LabelA)
    end
    
end
   for i=1:rowT
       temp1 = sum(-1*resultLabels(rowS+rowA+i,ceil(N/2):N).*betaT())
       temp2 = -1/2*sum(log(betaT(ceil(N/2):N)))
       if(temp1>temp2)
           H(i,1) = 1;
       else
           H(i,1) =-1;
       end
   end
end