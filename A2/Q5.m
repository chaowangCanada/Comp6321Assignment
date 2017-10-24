function Q5()

%a, use the iterative recursive least-squares: Newton-Raphson for logistic regression
X = load('wpbcx.dat');
Y = load('wpbcy.dat');

extendX = [X, ones(size(X, 1),1)];
W_int = 0.2*ones( size(extendX,2), 1 );  %initial guess
J(extendX, Y, W_int);

W = NewtonRaphson(extendX, Y, W_int, 20);

%b Gaussian naive Bayes classifier,


function W=NewtonRaphson(X, Y, W_int,iter)
%init phi
W = W_int;
Phi = X;

R = eye(length(X));

for i=1:iter
    for j = 1:length(X)
        R(j, j) = sigmoid(X(j,:)*W)*(1-sigmoid(X(j,:)*W));
    end
    W = pinv(Phi'*R*Phi)*Phi'*R*(Phi*W - pinv(R)*(sigmoid(Phi*W)-Y));
    J(X, Y, W);
end

end

function errorF = J(X, Y, W)
errorF = -sum(Y.*log(sigmoid(X*W)) + (1-Y).*log(1 - sigmoid(X*W)));
end

function logFunc = sigmoid(x)
logFunc = 1./(1+exp(-x));
end

end

