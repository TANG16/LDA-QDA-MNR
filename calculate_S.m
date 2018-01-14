function S = calculate_S(X, W)
%CALCULATE_S calculate a matrix of probabilities for each class
%            Each class contains numbers of samples

% input arguments
%   X: a matrix of N_train by D training samples
%   W: a matrix of D by C; where D and C are # of features and classes
%      contains random weights at the beginning

% author: Maryam Najafi
% created date: Oct 27, 2016

for k = 1 : 3
    for i = 1 : size(X,1)
        numerator = exp(W(:, k)' * X(i,:)');
        denominator = sum(exp(W(:, 1)' * X(i,:)') + exp(W(:, 2)' * X(i,:)')...
            + exp(W(:, 3)' * X(i,:)'));
        
        S(i , k) = numerator / denominator;

    end
end
end