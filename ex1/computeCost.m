function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = theta'*X'; % theta' is 1xn+1 and X' is n+1xm then h is 1xm where h_i = 0_0 + x_i*0_1
diff = h - y'; % y is mx1 and h is 1xm then y' is 1xm and diff is 1xm
cuad = diff.^2;
sumCuads = sum(cuad);
J = sumCuads/(2*m);

% =========================================================================

end
