function [TrainData,TrainDataLabel,TestData,TestDataLabel]= GetTrainTestData(Data,DataLabel,Percent)

%TrainData 随机选择的训练数据
%TraninDataLabel 训练数据的标签
%TestData 随机选择的测试数据
%TestDataLabel 测试数据的标签
%Percent 训练数据所占的百分比
%Data 所有数据
%DataLabel 数据标签

TrainDataNum = ceil(size(Data,1)*Percent);
SelectDataIndex = randperm(size(Data,1));

TrainData = Data(SelectDataIndex(1:TrainDataNum),:);
TrainDataLabel = DataLabel(SelectDataIndex(1:TrainDataNum));

TestData = Data(SelectDataIndex(TrainDataNum+1:size(Data,1)),:);
TestDataLabel = DataLabel(SelectDataIndex(TrainDataNum+1:size(Data,1)));
end