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

% Theta1: 25 x 401
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta2: 10 x 26 
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
 
% Setup some useful variables m: number of training examples
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% --forward propagation to compute cost function including regularization--

% X: 5000 x 400
X = [ones(m,1) X];
% X: 5000 x 401

a1 = X; %a1: 5000 x 401

z2 = a1 * Theta1'; %z2: 5000 x 25
a2 = sigmoid(z2); % 5000 x 25
a2 = [ones(size(a2,1),1) a2]; % a2 = 5000 x 26

z3 = a2 * Theta2'; %a3: 5000 x 10
a3 = sigmoid(z3); % 5000 x 10

% turn y into a matrix - y_matrix: 5000 x 10
y_matrix = eye(num_labels)(y,:); 

% apply the cost function formula to get regularized cost 
% (if unregularized just set lambda = 0)
% sum all the outputs(y) and inputs(X, represented as a3) together
J = -1/m * ( sum(sum( y_matrix.* log(a3))) + ...
             sum(sum((1-y_matrix) .* log(1-a3)))) + ...
    lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + ... % regularization excludes Theta1/2 bias units
                    sum(sum(Theta2(:,2:end).^2)));
 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% --backpropagation to compute gradient including regularization--

% difference btwn forwardprop output and y
d3 = (a3 - y_matrix); %5000 x 10

% backpropagation δ: d(n) = d(n+1)*Theta(n) excluding bias unit, 
%                    then scaled by g'(z2)
d2 = (d3 * Theta2(:,2:end)).*sigmoidGradient(z2); %5000 x 10 * 10 x 25 -> 5000 x 25

% backpropagation ∆: Delta(n) = sum(a(n)'* d(n+1))
Delta1 = (d2' * a1); %25x5000 * 5000x401 -> 25x401
Delta2 = (d3' * a2); %10x5000 * 5000x26 -> 10x26

% take out Theta bias unit for regularization
Theta1_reg = [zeros(rows(Theta1),1) Theta1(:, 2:end)];
Theta2_reg = [zeros(rows(Theta2),1) Theta2(:, 2:end)];

% regularized gradient (lambda = 0 if unregularized)
Theta1_grad = (Delta1 + lambda * Theta1_reg) / m; %25x401
Theta2_grad = (Delta2 + lambda * Theta2_reg) / m; %10*26


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
