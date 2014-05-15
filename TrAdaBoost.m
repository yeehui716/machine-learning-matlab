function [H] = TrAdaBoost(TrainS,TrainA,LabelS,LabelA,Test,N)
%
% H �������������շ�����ǩ
% TrainS ԭѵ������
% TrainA ����ѵ������
% LabelS ԭѵ�����ݱ�ǩ(������)
% LabelA ����ѵ�����ݱ�ǩ(������)
% Test  ��������
% N ��������
%%%%%%%%%%%%%% Learner ����������
% Write by ChenBo 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainData  = [TrainS;TrainA]
trainLabel = [LabelS;LabelA]

% rowS ԭѵ�������ĸ���, columnS ԭѵ��������ά�� ;  rowA ����ѵ�������ĸ���, columnA ����ѵ��������ά��
[rowS,columnS] = size(TrainS)
[rowA,columnA] = size(TrainA)
%rowT ���������ĸ��� , columnT����������ά��
[rowT,columnT] = size(Test)

%�㷨��Ҫ��������ѵ�����������ݵ�Ԥ������
testData = [trainData;Test]

%��ʼ��ѵ������Ȩ�ؼ�Beta
weight = ones(rowS+rowA,1)/(rowS+rowA)
beta  = 1/(1+sqrt(2*log(rowA/N)))
betaT = zeros(1,N)

%��¼N�ε���ѵ���Ľ���������ѵ���������������ݣ�
resultLabels = ones(rowS+rowA+rowT,N)

for i=1:N
    %p = weight./sum(weight)
    %resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,5, weight);
    resultLabels(:,i)= WeightedKNN(trainData,trainLabel,testData,5);
    er = ErrorRate(LabelS,resultLabels(1:rowS,i),weight(1:rowS))  %ԭѵ�����ݵķ���������
    if(er>0.5)
        er = 0.5
    end
    if(er==0)
        er=0.001
    end
    betaT(1,i)=er/(1-er)
    for j=1:rowS %����ԭѵ�����ݵ�Ȩ��
%         temp1 = resultLabels(j,i)-LabelS(j)
%         temp2 = abs(resultLabels(j,i)-LabelS(j))
%         temp3 = beta^abs(resultLabels(j,i)-LabelS(j))
%         temp4 = beta^temp2
        weight(j) = weight(j)* betaT(i)^abs(resultLabels(j,i)-LabelS(j))
    end
    for j=1:rowA %���¸���ѵ�����ݵ�Ȩ��
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
