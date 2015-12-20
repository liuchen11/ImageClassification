clear;close all;clc;

load('train.mat');

train.y       = double(train.y);
train.X_cnn   = double(train.X_cnn);
train.X_hog   = double(train.X_hog);

y_train = train.y - 4;

for i = 1:length(y_train)
    if y_train(i) 
        y_train(i) = 1;
    end
end

X_train = train.X_hog;

instanceNum = size(X_train,1);
featureNum  = size(X_train,2);

times        = 10;
trainLoss    = zeros(times,9);
validateLoss = zeros(times,9);

rates    = 0.1:0.1:0.9;

for time = 1:times
    time
	for i = 1:size(rates,2)
		rate=rates(i);
		shuffle=randperm(instanceNum);
		shuffledX=X_train(shuffle,:);
		shuffledY=y_train(shuffle,:);
		trainInstanceNum=int32(instanceNum*rate);

		trainX=shuffledX(1:trainInstanceNum,:);
		validateX=shuffledX(trainInstanceNum+1:instanceNum,:);
		trainY=shuffledY(1:trainInstanceNum);
		validateY=shuffledY(trainInstanceNum+1:instanceNum);

		normtrainX=(trainX-ones(size(trainX,1),1)*mean(trainX))./...
				(ones(size(trainX,1),1)*std(trainX));
		normvalidateX=(validateX-ones(size(validateX,1),1)*mean(trainX))./...
				(ones(size(validateX,1),1)*std(trainX));

		trainX=[ones(trainInstanceNum,1),normtrainX];
		validateX=[ones(instanceNum-trainInstanceNum,1),normvalidateX];
		beta=logisticRegression(trainY,trainX,0.001);
		trainLoss(time,i)=computeCost(trainY,trainX,beta);
		validateLoss(time,i)=computeCost(validateY,validateX,beta);
		meg=sprintf('%d trials with training radio %f',time,rate);
		disp(meg);
	end
end

trainLossMean=mean(trainLoss(:,1:9));
validateLossMean=mean(validateLoss(:,1:9));
disp(trainLossMean);
disp(validateLossMean);
figure;
plot(rates,trainLossMean,'r','linewidth',2);
hold on;
plot(rates,validateLossMean,'b','linewidth',2);
hold on;
% plot(rates,trainLoss,'color',[0.3,0,0]);
% hold on;
% plot(rates,validateLoss,'color',[0,0,0.3]);