function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = theta'*X';

sigh = sigmoid(h');

diff = sigh - y;

logh = log(sigh);

log1h = log(1.- sigh);

y1 = 1.-y;

s = y.*logh .+ y1.*(log1h);

cuad = theta.^2;

cuad(1, 1) = 0;

J = -1/m*sum(s) + (lambda/(2*m))*sum(cuad, 1);

cantF = size(X, 2);

prod = zeros(size(X));
for j = 1:cantF
	prev = diff .* X(:, j);
	prod(:, j) = prev;
end

opt1 = sum(prod, 1)';  % (1xn+1)' = n+1x1

mult = lambda.*theta;
mult(1,1) = 0;

opt1 = opt1 .+ mult;

grad = (1/m).*opt1;

% =============================================================

end
