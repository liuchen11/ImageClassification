% -------------------------------------------------------------------
% This file can be used to classify images into four classes
% using naive logistic regression
% 5 fold cross validation is carried out
% Change the feature name when loading different data set
% Output: the prediction matrix and BER
%--------------------- C. Liu & M. Zhao -----------------------------

clearvars; close all; clc

load ../train/train.mat
train.X_cnn   = double(train.X_cnn);
train.X_hog   = double(train.X_hog);
train.y       = double(train.y);

trainX    = train.X_hog; % change the feature name if using cnn
trainY    = train.y;

% using 5 fold cross validation
K   = 5;
idx = randperm(length(train.y));
Nk  = floor(length(train.y)/K);

for k = 1:K
	idxCV(k,:) = idx(1+(k-1)*Nk:k*Nk);
end

Y = train.y;
X = train.X_hog; % change the feature name if using cnn

classes   = unique(train.y);
numClass  = numel(classes);


% classes and corresponding labels:
% 1 ===>  Airplane
% 2 ===>  Car
% 3 ===>  Horse
% 4 ===>  Others
% Voting logistic

for k = 1:K
	
    idxTe = idxCV(k,:);
    idxTr = idxCV([1:k-1 k+1:end],:);
	idxTr = idxTr(:);
	Yte   = trainY(idxTe);
	Xte   = trainX(idxTe,:);
	Y     = trainY(idxTr);
	X     = trainX(idxTr,:);
    
 	tXTe  = [ones(size(Xte,1),1) Xte];
    
% ------- Preprocessing the trainin data -------------
trainCell             = cell(1,6); 

train12idx = find(Y(:) == 1 | Y(:) == 2);
train13idx = find(Y(:) == 1 | Y(:) == 3);
train14idx = find(Y(:) == 1 | Y(:) == 4);
train23idx = find(Y(:) == 2 | Y(:) == 3);
train24idx = find(Y(:) == 2 | Y(:) == 4);
train34idx = find(Y(:) == 3 | Y(:) == 4);

label12    = abs(Y(train12idx) - 2)./1;
label13    = abs(Y(train13idx) - 3)./2;
label14    = abs(Y(train14idx) - 4)./3;
label23    = abs(Y(train23idx) - 3)./1;
label24    = abs(Y(train24idx) - 4)./2;
label34    = abs(Y(train34idx) - 4)./1;

trainCell{1,1}        = [X(train12idx,:) label12];
trainCell{1,2}        = [X(train13idx,:) label13];
trainCell{1,3}        = [X(train14idx,:) label14];
trainCell{1,4}        = [X(train23idx,:) label23];
trainCell{1,5}        = [X(train24idx,:) label24];
trainCell{1,6}        = [X(train34idx,:) label34];
% ------------------------------------------------------

% Construct 6 logistic classifiers

logModels = cell(1,6);
rng(1); 


for j = 1:6  
    logModels{1,j} = logisticRegression(trainCell{1,j}(:,end),[ones(size(trainCell{1,j},1),1) trainCell{1,j}(:,1:end-1)],0.1);
end

label = [];

for j = 1:6   
    predictLog = sigmoid(tXTe*logModels{1,j});
    [label(:,j),~] = round(predictLog);    
end



% start the voting process

for i = 1:length(Yte)
    i
    voting = zeros(1,4);

    for j = 1:6;
        
        switch j
            
            case {1} 
                
                if label(i,j) 
                    voting(1,1) = voting(1,1) + 1;
                else
                    voting(1,2) = voting(1,2) + 1;
                end
                     
            case {2} 
               
                if label(i,j)  
                    voting(1,1) = voting(1,1) + 1;
                else
                    voting(1,3) = voting(1,3) + 1;
                end
                
            case {3} 
                
                if label(i,j)  
                    voting(1,1) = voting(1,1) + 1;
                else
                    voting(1,4) = voting(1,4) + 1;
                end
                
            case {4} 
                
                if label(i,j)  
                    voting(1,2) = voting(1,2) + 1;
                else
                    voting(1,3) = voting(1,3) + 1;
                end
                
            case {5} 
                
                if label(i,j)  
                    voting(1,2) = voting(1,2) + 1;
                else
                    voting(1,4) = voting(1,4) + 1;
                end
                
            case {6} 
                
                if label(i,j)  
                    voting(1,3) = voting(1,3) + 1;
                else
                    voting(1,4) = voting(1,4) + 1;
                end
        end
        
        [~,pred(i)] = max(voting);
                
    end
end

pred = pred';

nnPred = [1*(pred == 1), ...
          1*(pred == 2), ...
          1*(pred == 3), ...
          1*(pred == 4) ]; 

[matrix,BER] = balancedErr(nnPred,Yte);

out(k) = BER;
pred = zeros(1,length(pred));
end

mean(out)



