function [H] = TrAdaBoost(TrainS,TrainA,LabelS,LabelA,Test,N)
%
% H 测试样本分类结果
% TrainS 原训练样本
% TrainA 辅助训练样本
% LabelS 原训练样本标签
% LabelA 辅助训练样本标签
% Test  测试样本
% N 迭代次数
%%%%%%%%%%%%%% Learner 基本分类器
% Write by ChenBo 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData  = [TrainS;TrainA]
trainLabel = [LabelS;LabelA]

[rowS,columnS] = size(TrainS)
[rowA,columnA] = size(TrainA)

[rowT,columnT] = size(Test)


testData = [trainData;Test]

%初始化weights 
weight = ones(rowS+rowA,1)/(rowS+rowA)
beta  = 1/(1+sqrt(2*log(rowA/N)))
betaT = zeros(1,N)

%由于迭代与最终计算需要使用所有样本标签
resultLabels = ones(rowS+rowA+rowT,N)

for i=1:N
    %p = weight./sum(weight)
    %resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,5, weight);
    resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,5);
    er = ErrorRate(LabelS,resultLabels(1:rowS,i),weight(1:rowS))      
    if(er>0.5)
        er = 0.5
    end
    if(er==0)
        er=0.001
    end
    betaT(1,i)=er/(1-er)
    for j=1:rowS %调整源域训练样本权重
%         temp1 = resultLabels(j,i)-LabelS(j)
%         temp2 = abs(resultLabels(j,i)-LabelS(j))
%         temp3 = beta^abs(resultLabels(j,i)-LabelS(j))
%         temp4 = beta^temp2
        weight(j) = weight(j)* betaT(i)^abs(resultLabels(j,i)-LabelS(j))
    end
    for j=1:rowA %调整辅助训练样本权重
        weight(rowS+j) = weight(rowS+j)*beta^(-abs(resultLabels(rowS+j,i))-LabelA(j))
    end
    
end
for i=1:rowT
    temp1 = sum(resultLabels(rowS+rowA+i,ceil(N/2):N).*log(1./betaT(1,ceil(N/2:N))))
    temp2 = 1/2*sum(log(1./(betaT(1,ceil(N/2):N))))
    if(temp1>=temp2)
        H(i,1) = 1;
    else
        H(i,1) = 0;
    end
end
end
