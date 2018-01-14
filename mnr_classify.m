function [Lpred, Scores] = mnr_classify(Xtest, Weights)
% MNR_CLASSIFY classify the test dataset using Multinomial Reg. classifier

% input arguments
%   Xtest: 1 sample (1 by d) where d = 2 number of features
%   Weights: matrix of d by C; where d = 2 and C = 3 num of classes

% output params
%   Lpred: predicted label which is a number between 1 to 3
%   Scores: matrix of size a 1 by C
%           the posterior probability of given sample regarding each class

% author: Maryam Najafi
% created date: Oct 27, 2016
% last date modified: Oct 30, 2016

N_test = size(Xtest,1);
Lpred = zeros(N_test, 1);

Xtest = [ones(N_test,1) Xtest];

Scores = Xtest * Weights;
[a Lpred] = max(Scores,[],2);
% [m Lpred] = max(Scores);
        
% for k = 1 : 3
%     for i = 1 : N_test
%         numerator = exp(Weights(:, k)' * Xtest(i,:)');
%         denominator = sum(exp(Weights(:, 1)' * Xtest(i,:)') + exp(Weights(:, 2)' * Xtest(i,:)')...
%             + exp(Weights(:, 3)' * Xtest(i,:)'));
        
%         Scores(i , k) = numerator / denominator;
%         [m Lpred(i)] = max(Scores(i,:));

%     end
% end


end