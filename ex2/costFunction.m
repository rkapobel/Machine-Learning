function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

h = theta'*X';

sigh = sigmoid(h');

diff = sigh - y;

logh = log(sigh);

log1h = log(1.- sigh);

y1 = 1.-y;

s = y.*logh .+ y1.*(log1h);

J = -1/m*sum(s);

cantF = size(X, 2);

prod = zeros(size(X));
for j = 1:cantF
	prev = diff .* X(:, j);
	prod(:, j) = prev;
end

grad = ((1/m)*sum(prod, 1))';  % (1xn+1)' = n+1x1

% =============================================================

end
