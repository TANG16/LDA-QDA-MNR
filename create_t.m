function T = create_t(Xtrain, Ltrain)
% CREATE_T encode each sample's label onto a vector t

% input
%   Xtrain: N_train by d matrix (training samples)
%   Ltrain: N_train by 1 containing the label of each sample

% output
%   T: N_train by C matrix; where C is the total number of classes (3)

% author: Maryam Najafi
% date: Oct 28, 2016

N_train = size(Ltrain, 1);
C = 3;  % num of classes

T = zeros(N_train, C);

for i = 1 : N_train
    T(i,Ltrain(i)) = 1;
end

end