function [Lpred, Scores] = da_classify(Xtest, Means, Covariances, Priors)
%DA_CLASSIFY Utilizes QDA/LDA classifiers to classify iris dataset.

%   input arguments
%   Xtest: a matrix of size N_test by D, containing testing samples
%   Means: a matrix of size k by D; where k is number of classes. The row k
%          corresponds to the estimated mean of kth class
%   Covariances: a matrix of size DxDxC that contains DxD estimated 
%                covariance matrices for each class C
%   Priors: a vec of size k containing each one of classes' estimate priors

%   output params
%   Lpred: a vec of size N_test whose elements are predicted labels
%   Scores: a matrix of size N_test by C; each row is three posterior class
%           probabilities corresponding to each class ( p(w_i|X) )

%   The estimated posterior probability for class 1 is given by:
%   p(w_1 | x) = (f(x | w_1) * p_1)/f(x)
%   Compare every pair-class and find the champion among the classes.

D = 2; % num of features Sepal length and width
N = size(Xtest,1);
for n = 1:N
    post_1(n) =  (exp ((-1/2) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(1,:)) * ...
        inv(Covariances(:,:,1)) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(1,:))' ))...
        / sqrt((2 * pi)^D) * det(Covariances(:,:,1)) * Priors(1);

end

for n = 1:N
    post_2(n) =  (exp ((-1/2) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(2,:)) * ...
        inv(Covariances(:,:,2)) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(2,:))' ))...
        / sqrt((2 * pi)^D) * det(Covariances(:,:,2)) * Priors(2);

end


for n = 1:N
    post_3(n) =  (exp ((-1/2) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(3,:)) * ...
        inv(Covariances(:,:,3)) * (Xtest(n,:) - ones(size(Xtest(n,:), 1), 1) * Means(3,:))' ))...
        / sqrt((2 * pi)^D) * det(Covariances(:,:,3)) * Priors(3);

end

Scores(:,1) = post_1;
Scores(:,2) = post_2;
Scores(:,3) = post_3;

Lpred = zeros(1,N);
for n = 1:N
    [m, I] = max(Scores(n,:));
    Lpred(n) = I;
end




end

