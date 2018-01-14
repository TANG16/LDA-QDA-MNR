function W = generateRandWeights(D, C)
% GENERATERANDWEIGHTS generates a matrix of DxC random numbers

% input arguments
%   D: 2 features
%   C: 3 classes

% output params
%   W: a matrix of size D by C of random numbers between 0 and 1

W = ones(D, C) .* rand(D, C);

end