function [H] = TrAdaBoost(TrainS,TrainA,LabelS,LabelA,Test,N)
%
% H 测试样本的最终分类标签
% TrainS 原训练数据
% TrainA 辅助训练数据
% LabelS 原训练数据标签(列向量)
% LabelA 辅助训练数据标签(列向量)
% Test  测试数据
% N 迭代次数
%%%%%%%%%%%%%% Learner 基本分类器
% Write by ChenBo 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData  = [TrainS;TrainA]
trainLabel = [LabelS;LabelA]

% rowS 原训练样本的个数, columnS 原训练样本的维数 ;  rowA 辅助训练样本的个数, columnA 辅助训练样本的维数
[rowS,columnS] = size(TrainS)
[rowA,columnA] = size(TrainA)
%rowT 测试样本的个数 , columnT测试样本的维数
[rowT,columnT] = size(Test)

%算法需要计算所有训练、测试数据的预测结果
testData = [trainData;Test]

%初始化训练样本权重及Beta
weight = ones(rowS+rowA,1)/(rowS+rowA)
beta  = 1/(1+sqrt(2*log(rowS/N)))
betaT = zeros(1,N)

%记录N次迭代训练的结果（包括训练数据与测试数据）
resultLabels = ones(rowS+rowA+rowT,N)

for i=1:N
    %p = weight./sum(weight)
    %resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,5, weight);
    resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,10);
    er = ErrorRate(LabelS,resultLabels(1:rowS,i),weight(1:rowS))  %原训练数据的分类错误率
    if(er>0.5)
        er = 0.5
    end
    if(er==0)
        er=0.001
    end
    betaT(1,i)=er/(1-er)
    for j=1:rowS %更新原训练数据的权重
        weight(j) = weight(j)*beta^abs(resultLabels(j,i)-LabelS(j))
    end
    for j=1:rowA %更新辅助训练数据的权重
        weight(rowS+j) = weight(rowS+j)*betaT(i)^(-abs(resultLabels(rowS+j,i))-LabelA(j))
    end
    
end
resultLabels(:,1)
resultLabels(:,N)
a = sum(resultLabels(:,1))
a = sum(resultLabels(:,N))
betaT(1,:)
for i=1:rowT
    temp1 = sum(resultLabels(rowS+rowA+i,ceil(N/2):N).*log(1./betaT(ceil(N/2:N))))
    temp2 = 1/2*sum(log(1./(betaT(ceil(N/2):N))))
    if(temp1>=temp2)
        H(i,1) = 1;
    else
        H(i,1) = 0;
    end
end
end