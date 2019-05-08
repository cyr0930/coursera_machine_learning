function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
z = X * theta;
h = sigmoid(z);
theta1 = theta(2:end);
J = 1/m* (-y'*log(h) + (y'-1)*log(1-h) + lambda/2*sum(theta1.^2));
grad = 1/m * X'*(h-y);
grad(2:end) += lambda/m*theta1;
% =============================================================
end
