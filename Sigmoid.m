function sigma = Sigmoid(W,x)
%Expects W to be a column vector to be transposed to row vector
%Expects x to be a column vector
    sigma = 1./(1 + exp(-W'*x));
end