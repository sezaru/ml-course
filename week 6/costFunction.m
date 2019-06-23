function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for linear regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for linear regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

hypothesis = X * theta;

J = (1 / (2 * m)) * sum((hypothesis - y) .^ 2);
% OR
J = (1 / (2 * m)) * (hypothesis - y)' * (hypothesis - y);

grad = (1 / m) * X' * (hypothesis - y);

end