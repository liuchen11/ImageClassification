% -------------------------------------------------------------------
% This file can be used to classify images into two classes
% using naive logistic regression
% 5 fold cross validation is carried out
% Change the feature name when loading different data set
% Output: the prediction matrix and BER
%--------------------- C. Liu & M. Zhao -----------------------------
clearvars; close all; clc

load ../train.mat
train.X_cnn   = double(train.X_cnn);
train.X_hog   = double(train.X_hog);
train.y       = double(train.y);

% preprocessing the data:
% 0 ==> others
% 1 ==> Car, Plane, Horse
for i = 1:length(train.y)
    if train.y(i) == 4
        train.y(i) = 0;
    else
        train.y(i) = 1;
    end
end
% -----------------------

trainX    = train.X_cnn; % change the feature name if using cnn
trainY    = train.y;

K   = 5;
idx = randperm(length(train.y));
Nk  = floor(length(train.y)/K);

for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

Y = train.y;
X = train.X_cnn; % change the feature name if using cnn

classes   = unique(train.y);
numClass  = numel(classes);

matrixCell = cell(1,K);

for k=1:K
	
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
	idxTr = idxTr(:);
	Yte   = trainY(idxTe);
	Xte   = trainX(idxTe,:);
	Y     = trainY(idxTr);
	X     = trainX(idxTr,:);
    
    % add one column 1 to training and validation set
    tXTr = [ones(size(X,1),1) X];
	tXTe = [ones(size(Xte,1),1) Xte];
    
    % training beta 
    betaLog    = logisticRegression(Y,tXTr,.1);
	
    % making prediction
    predictLog = sigmoid(tXTe*betaLog);
    label      = round(predictLog);

% building the prediction matrix and calculating BER
 matrix = zeros(2,2);

for l = 1:length(Yte)
    
    switch Yte(l)
        case 1
            if label(l) == 1
                matrix(1,1) = matrix(1,1) + 1;
            else
                matrix(1,2) = matrix(1,2) + 1;
            end
        case 0
            if label(l) == 0
                matrix(2,2) = matrix(2,2) + 1;
            else
                matrix(2,1) = matrix(2,1) + 1;
            end
    end
end

matrixCell{1,k} = matrix;

BER = (matrix(1,1)/(matrix(1,1) + matrix(1,2)) + matrix(2,2)/(matrix(2,1) + matrix(2,2)))/2;

out(k) = BER;

end

mean(out)



