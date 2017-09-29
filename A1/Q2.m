function Q2()

x = load('hw1x.dat');
y = load('hw1y.dat');

scatter(x,y);

hold on;

extend = ones(size(x));
x_extend = [x extend];

for weight=1.0:1:50.0
  
  U = eye(size(x)(1));
  U(length(x),length(x))= U(length(x),length(x)) *weight;
  W = WeightedLinearRegression(x_extend, y, U);

  W_x = (min(x):0.1:max(x))';
  W_y = [W_x, ones(size(W_x))]*W;

  plot(W_x, W_y); 
end
% Q1.b linear regression
function W = WeightedLinearRegression(X, Y, U)
  W = pinv(X'*U*X)*X'*U*Y;
end

end