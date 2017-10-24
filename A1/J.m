
function errorF = J(X, Y, W)
errorF = -sum(Y.*log(sigmoid(X*W)) + (1-Y).*log(1 - sigmoid(X*W)));
end