function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% X: m*n (300x2) ; centroids: K*n(3x2)

% Set K
K = size(centroids, 1);



% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% initialize all the distances for all samples X to each k centroids, m*K
distance = zeros(size(X,1), K);

for k = 1:K
% get ||x(i) − μj||^2 for all X, and store it to the corresponding column (as each k) 
% sum(X,2) -> suming each row
  distance(:,k) = sum((X - centroids(k,:)).^2,2);
end

% get the minimun value (M) of each row and the corresponding index (I)
[M,I] = min(distance, [], 2); % m*2
idx = I;  % m*1



% =============================================================

end

