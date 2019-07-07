function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

errors = [];
for c_step = steps
  for sigma_step = steps
      model = svmTrain(X, y, c_step, @(x1, x2) gaussianKernel(x1, x2, sigma_step));
      predictions = svmPredict(model, Xval);
      errors = [errors; c_step, sigma_step, mean(double(predictions ~= yval))];
  endfor
endfor

[minValue, rowIndex] = min(errors(:, 3));
min_error = errors(rowIndex, :);

C = min_error(1);
sigma = min_error(2);

end