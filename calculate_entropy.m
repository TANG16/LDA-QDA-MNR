function E = calculate_entropy(T, S, C)
% CALCULATE_ENTROPY calculate the cross-entropy of t w.r.t. s

% input arguments
%   T: N_train by C matrix; where N_train and C are num of samples and
%   classes
%   S: N_train by 

% author: Maryam Najafi
% date: Oct 28. 2016
% last date modified: Oct 29, 2016


% E = - sum(diag(T' * log (S)));
E = - dot (T, log(S));
E = sum(E);
   
   
end