function d = calculate_gradient_CE (X, S, T)
% CALCULATE_GRADIENT_CE the gradient of the loss function (Cross-Entropy)

% input args
%   X: N_train by D training samples
%   S: N_train by C matrix full of probabilites of each sample w.r.t. 
%      C = {1, 2, 3}
%   T: N_train by C matrix containing {0,1}C

% output args
%   d: descent direction which is the gradient of CE


% author: Maryam Najafi
% date: Oct 28, 2016

d = X' * (S - T);


end