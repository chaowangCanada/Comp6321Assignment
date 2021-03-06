function Q1()

x = load('hw2x.dat');
y = load('hw2y.dat');
percentCut = 0.9;

x_training = x(1:round(length(x)*percentCut),:);
y_training = y(1:round(length(x)*percentCut),:);
x_test = x(round(length(x)*percentCut) + 1:length(x),:);
y_test = y(round(length(y)*percentCut) + 1:length(y),:);

%Q1.a 
figure 
hold on;
for lambda = 0:10:100
    W = L2Regulization(x_training, y_training, lambda);
%     ySol_training = ExpandX(x_training,1)*W;
%     ySol_test = ExpandX(x_test,1)*W;
    ySol_training = x_training*W;
    ySol_test = x_test*W;

    rMeanSqrEr = sqrt((sum((ySol_training - y_training).^2))/length(ySol_training));
    rMeanSqrEr_test = sqrt((sum((ySol_test - y_test).^2))/length(ySol_test));
    scatter(lambda, rMeanSqrEr,[],[0 0 1] );%blue
    scatter(lambda, rMeanSqrEr_test,[],[1 0 0]);  %Red
end

hold off;

figure
hold on ;
for lambda = 0:10:100
    W = L2Regulization(x_training, y_training, lambda);
    scatter(lambda, W(1),[],[0 0 1] );%blue
    scatter(lambda, W(2),[],[1 0 0]);  %Red
    scatter(lambda, W(3),[],[0 1 0]);  %GREEN
end


%Q1.C 
figure 
hold on;
for lambda = 0:10:100
    W = L1Regulization(x_training, y_training, lambda);
%     ySol_training = ExpandX(x_training,1)*W;
%     ySol_test = ExpandX(x_test,1)*W;
    ySol_training = x_training*W;
    ySol_test = x_test*W;

    rMeanSqrEr = sqrt((sum((ySol_training - y_training).^2))/length(ySol_training));
    rMeanSqrEr_test = sqrt((sum((ySol_test - y_test).^2))/length(ySol_test));
    scatter(lambda, rMeanSqrEr,[],[0 0 1] );%blue
    scatter(lambda, rMeanSqrEr_test,[],[1 0 0]);  %Red
end

hold off;

figure
hold on ;
for lambda = 0:10:100
    W = L1Regulization(x_training, y_training, lambda);
    scatter(lambda, W(1),[],[0 0 1] );%blue
    scatter(lambda, W(2),[],[1 0 0]);  %Red
    scatter(lambda, W(3),[],[0 1 0]);  %GREEN
end


function F = ExpandX(X,d)
  F = [X, ones(size(X, 1),1)];
  for n = 2:d
    F = [F(:,n-1).^n, F];
  end
end

function W = L2Regulization(X,Y, lambda)
  x_extend = X; %ExpandX(X,1);
  I = eye(size(x_extend, 2));
  W = pinv(x_extend'*x_extend + lambda.*I)*x_extend'*Y;
end

%Q1.b
function W = L1Regulization(X,Y, lambda)
  H = X'*X;
  f = -(Y'*X)';
  A = [ 1 1 1;
        1 1 -1;
        1 -1 1;
        -1 1 1;
        1 -1 -1;
        -1 1 -1;
        -1 -1 1;
        -1 -1 -1];
  b = lambda*ones(size(A,1),1);
  W = quadprog(H, f, A, b);
end


function J =errorF(X,Y,W,d, norm)
  if nargin < 5
    norm = 'no';
  end    
    
  x_expand = ExpandX(X,d);
    % normalize
  if strcmpi(norm, 'yes') ==1
    x_expand = Normalize(x_expand);
  end
  J = sum((x_expand*W - Y).^2)/2;
end

end


%
% hold on
% 
% %Q1.b plot linear regression
% 
% extend = ones(size(x));
% 
% x_extend = [x extend];
% 
% W = LinearRegression(x_extend, y);
% 
% W_x = (min(x):0.1:max(x))';
% W_y = [W_x, ones(size(W_x))]*W;
% 
% plot(W_x, W_y,'-r'); 
% 
% %Q 1.c, find error function
% error = sum(((x_extend*W - y).^2))/2;
% disp('Linear regression error function'), disp(error);
% 
% 
% % Q 1.d polynomial regression
% function W = PolyRegress(X,Y,d,norm)
%   if nargin < 4
%     norm = 'no';
%   end  
%     
%   x_extend = ExpandX(X,d);
%   % normalize
%   if strcmpi(norm, 'yes') ==1
%     x_extend = Normalize(x_extend);
%   end
%   W = pinv(x_extend'*x_extend)*x_extend'*Y;
% end
% 
% %Q1.e quadratic fit, and error function
% W = PolyRegress(x, y, 2);
% PlotPoly(x,y,W,2, '-g');
% x_expand = ExpandX(x,2);
% error = sum((x_expand*W - y).^2)/2;
% disp('quadratic regression error function'), disp(error);
% 
% %Q1.f cubic fit, and error function
% W = PolyRegress(x, y, 3);
% PlotPoly(x,y,W,3, '-m');
% x_expand = ExpandX(x,3);
% error = sum((x_expand*W - y).^2)/2;
% disp('cubic regression error function'), disp(error);
% 
% hold off
% title('linear,quadratic,cubic regression fit to data model')
% xlabel('x')
% ylabel('y')
% legend('scatter','linear', 'quadratic', 'cubic','Location','southeast')
% 
% %Q1.g explaination on report
% 
% %Q1.h five-fold cross validation
% kFoldValidation(x, y, 10, 5);
% 
% W = PolyRegress(x, y, 6, 'no');
% x_expand = ExpandX(x,6);
% error = sum((x_expand*W - y).^2)/2;
% disp('six deg poly regression error function'), disp(error);
% %plot best fit polinomial regression
% figure
% scatter(x,y)
% hold on
% PlotPoly(x,y,W,6, '-m')
% title('best fit 6 degree polinomial regression')
% xlabel('x')
% ylabel('y')
% 
% %Q1.i normalized five-fold cross validation
% disp('normalized: ');
% kFoldValidation(x, y, 10, 5, 'yes');
% 
% W = PolyRegress(x, y, 6, 'yes');
% x_expand = ExpandX(x,6);
% x_expand = Normalize(x_expand);
% error = sum((x_expand*W - y).^2)/2;
% disp('six deg normalized poly regression error function'), disp(error);
% %plot normalized best fit 
% figure
% scatter(Normalize(x),y)
% hold on
% PlotPoly(x_expand,y,W,6, '-m')
% title('best fit normalized 6 degree polinomial regression')
% xlabel('x')
% ylabel('y')
% 
% function kFoldValidation(X,Y,d,k,norm)
%     
%   if nargin < 5
%     norm = 'no';
%   end  
% 
%   if mod(length(X),k) == 0
%     numElem = length(X)/k;
%     Xgroup = zeros(numElem, 1, k);
%     Ygroup = zeros(numElem, 1, k);
%     
%     for n = 1:k
%       Xgroup(:, :, n) = X((n-1)*numElem+1 : n*numElem, :);
%       Ygroup(:, :, n) = Y((n-1)*numElem+1 : n*numElem, :);
%     end  
%     
%     crossValidResult = zeros(d, 3);
%     for degree = 1 : d
%       error_train = zeros(k,1);
%       error_valid = zeros(k,1);
%       for m = 1:k
%         Dx_valid = Xgroup(:,:,m);
%         Dy_valid = Ygroup(:,:,m);
%         Dx_train = x;
%         Dx_train((m-1)*numElem+1 : m*numElem) = [];
%         Dy_train = y;
%         Dy_train((m-1)*numElem+1 : m*numElem) = [];
%         W = PolyRegress(Dx_train, Dy_train, degree, norm);
%         error_train(m) = errorF(Dx_train, Dy_train, W, degree, norm);
%         error_valid(m) = errorF(Dx_valid, Dy_valid, W, degree, norm);
%       end
%       crossValidResult(degree, :) = [degree mean(error_train) mean(error_valid)];
%     end 
%     disp(sprintf('Cross Validation %d',k)),disp(crossValidResult);
%     %plot the error
%     figure
%     plot(crossValidResult(:,1), crossValidResult(:,2),'b-o',crossValidResult(:,1), crossValidResult(:,3),'r-s')
%     title('training error, validation error figure')
%     xlabel('degree')
%     ylabel('error')
%     legend('training error','validation error')
%   end  
% end 
% 
% % Q1.b linear regression
% function W = LinearRegression(X, Y)
%   W = pinv(X'*X)*X'*Y;
% end
% 
% function norm = Normalize(X)
%   norm = X(:,:);
%   for i = 1:size(X,2);
%       norm(:,i) = X(:, i)./(max(abs(X(:,i))));
%   end
% end 
% 
% function F = ExpandX(X,d)
%   F = [X, ones(size(X))];
%   for n = 2:d
%     F = [F(:,n-1).^n F];
%   end
% end
% 
% function PlotPoly(X, Y, W, d, font)
%   gridX = (min(X):0.05:max(X))';
%   gridY = ExpandX(gridX, d);
%   plot(gridX, gridY*W, font);
% end
% 
% function J =errorF(X,Y,W,d, norm)
%   if nargin < 5
%     norm = 'no';
%   end    
%     
%   x_expand = ExpandX(X,d);
%     % normalize
%   if strcmpi(norm, 'yes') ==1
%     x_expand = Normalize(x_expand);
%   end
%   J = sum((x_expand*W - Y).^2)/2;
% end
% 
% end