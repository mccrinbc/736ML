function [basis,y] = makeBasisFunctions(data,means,varargin)

%Multivariance Gaussian Basis Function
[dataRow,~] = size(data);
[meanLength,numFeatures] = size(means);

covariance = 2*eye([numFeatures,numFeatures]);

%rectangular matrix
basis = ones(dataRow, meanLength + 1); %+1 to keep the bias term untouched.

for j = 1:(meanLength + 1)
    for ii = 1:dataRow
        if j == 1
            basis(ii,1) = 1;
        else
            basis(ii,j) = mvnpdf(data(ii,:)',means(j-1,:)',covariance); % -1 to compensate for bias
        end
    end
end

y = 0; %needs to be declated

if ~isempty(varargin)
    y = basis*varargin{1}; %estimate of the true labels
end


end