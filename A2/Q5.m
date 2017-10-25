function Q5()

%a, use the iterative recursive least-squares: Newton-Raphson for logistic regression
X = load('wpbcx.dat');
Y = load('wpbcy.dat');

extendX = [X, ones(size(X, 1),1)];
W_int = 0.2*ones( size(extendX,2), 1 );  %initial guess
J(extendX, Y, W_int);

W = NewtonRaphson(extendX, Y, W_int, 20);

%b Gaussian naive Bayes classifier,
numData = length(Y);
numClass1 = 0;
numClass0 = 0;
sumXClass1=0;
sumXClass0=0;
X_exp = X(:,1);
X_exp = [X_exp, ones(size(X_exp,1),1)];
sigma =  cov(X_exp');

for i = 1:length(X_exp)
    if((Y(i)==1))
        numClass1 = numClass1+1;
        sumXClass1=sumXClass1+X_exp(i,1);
    else
        numClass0 = numClass0 +1;
        sumXClass0=sumXClass0+X_exp(i,1);
    end
end

miu1=sumXClass1/numClass1;
miu0=sumXClass0/numClass0;

theta1=numClass1/numData;

result = bayesClassifier(theta1, miu0, miu1, sigma, X_exp)


%C Gaussian naive Bayes classifier for all input
numData = length(Y);
numClass1 = 0;
numClass0 = 0;
X_exp = [X, ones(size(X,1),1)];
sumXClass1= zeros(1,size(X_exp,2));
sumXClass0= zeros(1,size(X_exp,2),1);
sigma =  cov(X_exp');

for i = 1:length(X_exp)
    if((Y(i)==1))
        numClass1 = numClass1+1;
        sumXClass1=sumXClass1+X_exp(i,:);
    else
        numClass0 = numClass0 +1;
        sumXClass0=sumXClass0+X_exp(i,:);
    end
end

miu1=sumXClass1./numClass1;
miu0=sumXClass0./numClass0;

theta1=numClass1/numData;

resultALL = bayesClassifier(theta1, miu0, miu1, sigma, X_exp,'all')

function result = bayesClassifier(theta1, miu0, miu1, sigma, X_exp, option)
    
    if nargin < 6
        option = 'no';
    end    
   
    if strcmpi(option, 'all') ==1
       miu1 = repmat(miu1,length(X_exp),1);
       miu0 = repmat(miu0,length(X_exp),1);
    end
   
    a = -1/2*(X_exp-miu1)'*pinv(sigma)*(X_exp-miu1) + 1/2*(X_exp-miu0)'*pinv(sigma)*(X_exp-miu0)+log(theta1/(1-theta1));
    result = 1./(1 + exp(-a));
end


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

