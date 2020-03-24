close all;
clear all;

crabdata = readmatrix('crabdata.csv');
data = crabdata; %make a copy 

classes = data(:,1); 
classes = classes - 1; %make the classes [0,1]
sex = data(:,2);   
index = data(:,3);

%remove the unneeded columns
data(:,1:3) = [];
 
[entries,features] = size(data);
for ii = 1:features
    data(:,ii) = rescale(data(:,ii),-1,1); %rescale all the features between [-1,1]
end

samples = [1:1:entries];
samples = samples(randperm(length(samples))); 

trainingSample = samples(1:ceil(0.75*length(samples))); % 75% of reshuffle is for training
testingSample  = samples(ceil(0.75*length(samples))+1:end); % remaining 25% used for testing

trainSet = data([trainingSample],:); trainClasses = classes(1:ceil(0.75*length(samples)));
testSet  = data([testingSample],:);  testClasses  = classes(ceil(0.75*length(samples))+1:end);

%like before, we're going to use a set of non-linear, gaussian basis
%functions with random means centered at training points. Covariance matrix
%is identity. Using 5 Basis functions as a start. 

W = zeros(features,1); %initalize the weight vector to zeros, +1 for the bias
basisNum = 4; % M-1 true basis functions 
means = trainSet(1:basisNum,:);
EPOCHS = 4;
RMSarrayTest = zeros(1,EPOCHS);
RMSarrayTrain = zeros(1,EPOCHS);
Ws = [];

for EPOCH = 1:EPOCHS %Using Training set 5 times
    [basis,R,y] = makeBasisR(trainSet,means,W,basisNum);
    W = W - inv(basis'*R*basis)*basis'*(y-trainClasses);
    Ws = [Ws, W];
    
    testEstimate = testSet*W;
    testEstimate = 1./(1 + exp(-testEstimate)); %passs through sigmoid to classify
    
    RMSarrayTest(1,EPOCH)  = sqrt(mean((testEstimate-testClasses).^2));
    RMSarrayTrain(1,EPOCH) = sqrt(mean((y-trainClasses).^2));
end

CM = zeros(2,2);
testEstimate = int8(testEstimate);
for ii = 1:length(testClasses)
    if testClasses(ii) == 1 & testEstimate(ii) == 1
        CM(1,1) = CM(1,1) + 1;
    elseif testClasses(ii) == 1 & testEstimate(ii) == 0
        CM(1,2) = CM(1,2) + 1;
    elseif testClasses(ii) == 0 & testEstimate(ii) == 1
        CM(2,1) = CM(2,1) + 1;
    elseif testClasses(ii) == 0 & testEstimate(ii) == 0
        CM(2,2) = CM(2,2) + 1; 
    else 
        print('Logical Statement Not Satisfied')
        ii
        break
    end
end

CM
misclassrate = (CM(1,1) + CM(2,2))/sum(sum(CM))

figure(1)
plot(1:EPOCHS, RMSarrayTest,'-o','MarkerIndices',1:EPOCHS,'Color',[0,0.7,0.9])
xlabel('EPOCH NUMBER','FontSize',16)
ylabel('RMS Error','FontSize',16)
title('Testing: RMS Error vs EPOCH Testing','FontSize',16)

figure(2)
plot(1:EPOCHS, RMSarrayTrain,'-o','MarkerIndices',1:EPOCHS,'Color',[0.5,0.2,0.4])
xlabel('EPOCH NUMBER','FontSize',16)
ylabel('RMS Error','FontSize',16)
title('Training: RMS Error vs EPOCH','FontSize',16)



%%%%%%%%%%%%%% Support Vector Implementation %%%%%%%%%%%%%%%
SVMmodel = fitcsvm(trainSet,trainClasses); %train the model
[labelSVM,score] = predict(SVMmodel,testSet); %classify testing points

CMsvm = zeros(2,2);
for ii = 1:length(testClasses)
    if testClasses(ii) == 1 & labelSVM(ii) == 1
        CMsvm(1,1) = CMsvm(1,1) + 1;
    elseif testClasses(ii) == 1 & labelSVM(ii) == 0
        CMsvm(1,2) = CMsvm(1,2) + 1;
    elseif testClasses(ii) == 0 & labelSVM(ii) == 1
        CMsvm(2,1) = CMsvm(2,1) + 1;
    elseif testClasses(ii) == 0 & labelSVM(ii) == 0
        CMsvm(2,2) = CMsvm(2,2) + 1; 
    else 
        print('Logical Statement Not Satisfied')
        ii
        break
    end
end

CMsvm
