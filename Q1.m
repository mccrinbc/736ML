close all;
clear all;
% Data Information
% 1. mpg: continuous 
% 2. cylinders: multi-valued discrete 
% 3. displacement: continuous 
% 4. horsepower: continuous 
% 5. weight: continuous 
% 6. acceleration: continuous 
% 7. model year: multi-valued discrete 
% 8. origin: multi-valued discrete 
% 9. car name: string (unique for each instance)

% Columns to Normalize: 3, 4, 5, 6
% Columns to Remove:    9
% Predicting MPG

data = readtable('auto-mpg.dat');
data = data(:,1:end-1); %remove the text column 

t = table2array(data(:,1)); %target value
train_labels = t(1:100,1); %testing target values
test_labels = t(101:end,1); %testing target values
data(:,1) = []; %remove target value

data = table2array(data);

dataNorm = data;
dataNorm(:,2) = normalize(data(:,2));
dataNorm(:,3) = normalize(data(:,3));
dataNorm(:,4) = normalize(data(:,4));
dataNorm(:,5) = normalize(data(:,5));

%trying to normalize columns 1 and 7
dataNorm(:,1) = normalize(data(:,1));
dataNorm(:,7) = normalize(data(:,7));


basisLength = [90];
lambda = [0,0.01,0.1,1,10,100,1000];
training_set = dataNorm(1:100,:);
testing_set = dataNorm(101:end,:);

RMSerrorTrain = [];
RMSerrorTest = [];

%from what I understand, we're going to have a Gaussian basis where each of
%the sigmas are equal within the set and the mean is a vector in the
%P-dimentional space that is randomly selected from the training set. 
for j = 1:length(lambda)
    lam = lambda(j);
    for ii = 1:length(basisLength)
        means = datasample(training_set,basisLength(ii),'Replace',false); %get a random sample of points for our means
        [basis,~] = makeBasisFunctions(training_set,means); %call external function
        
        [row,col] = size(basis); %only need this for the regularization term
        
        %These are the weights for the maximum likelihood
        weights = inv(lam.*eye(col) + basis'*basis)*basis'*(train_labels);
        %we don't need to have a 'training step' because we determining the
        %weights through an inverse of our entire dataset. 
        
        [~,y_test] = makeBasisFunctions(testing_set,means,weights); %passing in testing set
        [~,y_train] = makeBasisFunctions(training_set,means,weights);
        
        RMS_test = sqrt((1/length(y_test))*sum((y_test - test_labels).^2));
        RMSerrorTest = [RMSerrorTest RMS_test];
        
        RMS_train = sqrt((1/length(y_train))*sum((y_train - train_labels).^2));
        RMSerrorTrain = [RMSerrorTrain RMS_train];
    end
end

reg = true;
if reg == true 
    figure();
    semilogx(lambda,RMSerrorTest,lambda,RMSerrorTrain,'-o','MarkerIndices',1:length(lambda))
    title('RMS Error Training and Testing Set','FontSize',16)
    xlabel('Lambda Used','FontSize',16)
    ylabel('RMS Error','FontSize',16)
    legend('Testing','Training')
else
    figure();
    hold on 
    plot(basisLength,RMSerrorTest,'-o','MarkerIndices',1:length(RMSerrorTest))
    plot(basisLength,RMSerrorTrain,'-o','MarkerIndices',1:length(RMSerrorTrain))
    title('RMS Error Training and Testing Set','FontSize',16)
    xlabel('Number of Basis Functions','FontSize',16)
    ylabel('RMS Error','FontSize',16)
    hold off
    legend('Testing','Training')
end




