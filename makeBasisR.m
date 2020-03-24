
function [basis,R,y] = makeBasisR(data, means, W, basisNum)
    [rows,feat] = size(data);
    basis = zeros(rows,basisNum); %initalize basis matrix
    basis = [ones(rows,1) basis]; %this adds a basis function of 1s
    R = zeros(rows,rows); %sqaure matrix, num rows = num instances. 
    y = zeros(rows,1);    %response variable
    
    for ii = 1:rows
        for j = 1:basisNum
            x = data(ii,:);
            transform = mvnpdf(x,means(j,:)); %Identity covariance 
            basis(ii,j+1) = transform; %index one higher
        end
        x = basis(ii,:); %x's that have gone through the basis functions
        R(ii,ii) = Sigmoid(W,x')*(1-Sigmoid(W,x')); 
        y(ii,1)  = Sigmoid(W,x'); %to classify between 0 and 1.
    end
    
end