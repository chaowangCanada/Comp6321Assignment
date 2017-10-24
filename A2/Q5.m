function Q5()

%In this code, I will adopt Newton-Raphson algorithm to find the weight that minimum the cost function J(w)
X = load('wpbcx.dat');
Y = load('wpbcy.dat');

extendX = [ ones(size(X, 1), X];
W_int = ones( size(extendX,2), 1 );

W = NewtonRaphson(extendX, Y, W_int, 20);

function res = sigmoid(x)
    res = 1./(1+exp(-x));
end

function W=NewtonRaphson(X, Y, W_int,iter)
    
    %init phi
    W = Winit;
    Phi = X;

    R = eye(length(X));

    for i=1:iter
        for j = 1:length(X)
            R(j, j) = sigmoid(X(j,:)*W)*(1-sigmoid(X(j,:)*W));
        end
        W = pinv(Phi'*R*Phi)*Phi'*R*(Phi*W - pinv(R)*(sigmoid(Phi*W)-Y));
        J(X, Y, W)
    end
end

function res = J(X, Y, W)
    res = -sum (Y.*log(sigmoid(X*W)) + (1.-Y).*log(1 .- sigmoid(X*W)));
end

end
