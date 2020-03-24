close all;
clear all;
rng default %For reproducibility 

train = readmatrix('musicdata.csv');
test  = readmatrix('musictestdata.csv');
test(:,1) = []; %remove row index column

trainLabel = train(:,1);
trainLabel = normalize(trainLabel - min(trainLabel));
train(:,1) = [];
testLabel = test(:,1);
testLabel = normalize(testLabel - min(testLabel));
test(:,1) = [];

trainNorm = normalize(train);
testNorm  = normalize(test) ; 

lambdas = linspace(0,2,1000);
RMS_error_train = [];
RMS_error_test = [];
numNonZero = [];

for ii = 1:length(lambdas)
    weights = lasso(trainNorm,trainLabel,'Lambda',lambdas(ii),'CV',10); %built in lasso function, col vector
    numNonZero = [numNonZero, length(nonzeros(weights))];
    
    y_train = trainNorm*weights;
    RMS_error = sqrt(sum((y_train - trainLabel).^2))./length(trainLabel);
    RMS_error_train = [RMS_error_train, RMS_error];
    
    y_test  = testNorm*weights; 
    RMS_error = sqrt(sum((y_test - testLabel).^2))./length(testLabel);
    RMS_error_test = [RMS_error_test, RMS_error];
end

figure(1);
hold on
plot(log(lambdas),RMS_error_train,'-o','MarkerIndices',1:length(lambdas));
plot(log(lambdas),RMS_error_test,'-o','MarkerIndices',1:length(lambdas));
xlabel('Log(\lambda)','FontSize',16)
ylabel('RMS Error','FontSize',16)
title('Training Error','FontSize',16)
legend('Train','Test')

figure(2);
plot(log(lambdas),RMS_error_test,'-o','MarkerIndices',1:length(lambdas));
xlabel('Log(\lambda)','FontSize',16)
ylabel('RMS Error','FontSize',16)
title('Testing Error','FontSize',16)

figure(3);
plot(lambdas,numNonZero,'-o','MarkerIndices',1:length(lambdas));
xlabel('\lambda','FontSize',16)
ylabel('Number of NonZero Parameter Coefficients','FontSize',16)
title('Non-Zero Coefficent Count vs. \lambda','FontSize',16)


value = min(RMS_error_test);
index = find(min(value) == RMS_error_test);

optimal_lambda = lambdas(index);
testRMS = RMS_error_test(index);
trainRMS = RMS_error_train(index);




