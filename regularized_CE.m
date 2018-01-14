function reg_E = regularized_CE( T, S, W, r )
%REGULARIZED_CE calculates the Cross-Entropy using added penalty term

% input arguments
%   T: N_train by C matrix; where N_train and C are num of samples and
%   classes
%   S: N_train by 
%   r: penalty parameter
%   W: matrix DXC weights for all classes

% author: Maryam Najafi
% date: Oct 30. 2016
% last date modified: Oct 30, 2016


reg_E = - sum(diag(T' * log (S))) + r * norm(W, 2)^2;

end

