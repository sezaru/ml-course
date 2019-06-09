function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

X1 = sigmoid(X * Theta1');

% Add ones to the X1 data matrix
X1 = [ones(m, 1) X1];

[_, p] = max(sigmoid(X1 * Theta2'), [], 2);
end