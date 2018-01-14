function d = calculate_gradient_regularizedCE( X, S, T, W, r )
% CALCULATE_GRADIENT_REGULARIZEDCE the gradient of the loss function
%                                  (reg. Cross-Entropy)

% input args
%   X: N_train by D training samples
%   S: N_train by C matrix full of probabilites of each sample w.r.t. 
%      C = {1, 2, 3}
%   T: N_train by C matrix containing {0,1}C
%   r: penalty parameter
%   W: matrix DxC weights for all classes

% output args
%   d: descent direction which is the gradient of CE


% author: Maryam Najafi
% date: Oct 30, 2016

d = X' * (S - T) + 2 * r * norm(W,2);

end

