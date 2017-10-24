function Q2()

x = load('hw1x.dat');
y = load('hw1y.dat');

scatter(x,y);

hold on;

extend = ones(size(x));
x_extend = [x extend];

% max_num=max(K(:)) 
% [X Y]=ind2sub(size(K),max_num)

U = eye(size(x,1));
W = WeightedLinearRegression(x_extend, y, U);
err = errorF(x,y,W,1);
disp('Linear Regression'), disp(err);

[maxVal, maxIndex] = max(x);

errorOutput = zeros(20,2);

for weight=1.0:1:20
  
  U(maxIndex,maxIndex)= weight;
  W = WeightedLinearRegression(x_extend, y, U);

  W_x = (min(x):0.1:max(x))';
  W_y = [W_x, ones(size(W_x))]*W;

  plot(W_x, W_y); 
  
  err = errorF(x,y,W,1);
  fprintf('Linear Regression on weight factor: %d, error is : %d \n',weight, err);
  errorOutput(weight,:) = [weight, err];
end
  figure
  plot(errorOutput(:,1), errorOutput(:,2));
  
function W = WeightedLinearRegression(X, Y, U)
  W = pinv(X'*U*X)*X'*U*Y;
end

function J =errorF(X,Y,W,d)
  x_expand = ExpandX(X,d);

  J = sum((x_expand*W - Y).^2)/2;
end

function F = ExpandX(X,d)
  F = [X, ones(size(X))];
  for n = 2:d
    F = [F(:,n-1).^n F];
  end
end

end