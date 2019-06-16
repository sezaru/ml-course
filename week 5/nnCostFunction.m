function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%%%%% Gradient computation

a1 = X;

% Add ones to the a1 data matrix
a1 = [ones(m, 1) a1];

z2 = a1 * Theta1';

a2 = sigmoid(z2);

% Add ones to the a2 data matrix
a2 = [ones(m, 1) a2];

z3 = a2 * Theta2';

a3 = sigmoid(z3);

hypothesis = a3;

%%%%% Cost function

y_size = size(y, 1);

recoded_y = zeros(y_size, num_labels);

for index = 1:y_size
    recoded_y(index, y(index)) = 1;
end

% Compute the J for each label
Js = ones(1, size(hypothesis, 1)) * (-recoded_y .* log(hypothesis) - (1 - recoded_y) .* log(1 - hypothesis));

% Sum all Js
J = (1 / m) * ones(1, num_labels) * Js';

%%%%% Regularization

t1 = Theta1(:, 2:end) .^ 2;
t1 = ones(1, size(t1, 1)) * t1;
t1 = ones(1, input_layer_size) * t1';

t2 = Theta2(:, 2:end) .^ 2;
t2 = ones(1, size(t2, 1)) * t2;
t2 = ones(1, hidden_layer_size) * t2';

regularization = (lambda / (2 * m)) * (t1 + t2);

J = J + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

for t = 1:m
    a1 = X(t, :)';
    a1 = [1; a1];
    
    z2 = Theta1 * a1;

    a2 = sigmoid(z2);
    a2 = [1; a2];

    z3 = Theta2 * a2;

    a3 = sigmoid(z3);
    
    delta3 = a3 - recoded_y(t, :)';
    
    delta2 = Theta2' * delta3 .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end);
    
    Theta1_grad = Theta1_grad + delta2 * a1';
    Theta2_grad = Theta2_grad + delta3 * a2';
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

Theta1_first_column = zeros(size(Theta1, 1), 1);

Theta1_without_bias = Theta1(:, 2:end);
Theta1_without_bias = [Theta1_first_column, Theta1_without_bias];

Theta2_first_column = zeros(size(Theta2, 1), 1);

Theta2_without_bias = Theta2(:, 2:end);
Theta2_without_bias = [Theta2_first_column, Theta2_without_bias];

Theta1_grad = (1 / m) * Theta1_grad + ((lambda / m) * Theta1_without_bias);
Theta2_grad = (1 / m) * Theta2_grad + ((lambda / m) * Theta2_without_bias);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
