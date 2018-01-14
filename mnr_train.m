function [Weights, CEvalues, status] = mnr_train(Xtrain, Ltrain, maxIter, lambdaMax, alpha, tol)
% MNR_TRAIN train input dataset using Multinomial Regression classifier

% input arguments
%   Xtrain: N_train by d matrix full of training data samples
%   Ltrain: vec of size N_train containing each sample corresponding label
%   maxIter: 100 which is the max possible backtracking iteration in GD
%   lambdaMax: step length which is greater than 0
%   alpha: a fraction between 0 and 1; step length gets affected by * alpha
%   tol: is a threshold > 0 that specifies the termination condition

% output params
%   Weights: 
%   CEvalues: 
%   status: 


% According to the question raining MNR should be performed via 
% Gradient Descent (GD) using backtracking as a line search. The maximum
% possible backtracking either way should not exceed 100 iterations. 
% The logistic regression model that is being used:
% e^(w_{k}' * x); where k is the kth class and x is a training sample

% author: Maryam Najafi
% created date: Oct 27, 2016

% random weights for three two-feature classes
D = 2; % features
C = 3; % # of classes
% tol = log10(tol);

% create weights
W = generateRandWeights(D+1, C); % a DxC matrix of randomly generated weights
                                 % 1 extra row for the bias
N = size(Xtrain,1);
Xtrain = [ones(N,1) Xtrain];

% calculate s_k
S = calculate_S(Xtrain, W);

% create T: to encode each sample's label onto t
T = create_t(Xtrain, Ltrain);

% update weights
[Weights, CEvalues, status] = GD_backtracking(Xtrain, Ltrain, lambdaMax, alpha, W, Xtrain, T, S, tol, maxIter);

end

function [W_0, E, stop] = GD_backtracking(Xtrain, Ltrain, lambdaMax, alpha, W_0, X, T, S, tol, maxIter)
num_of_iters = [];
big_i = 0;

C = 3; % # of classes

% figure();
% title(sprintf('CE w.r.t. number of iterations for maxLambda = %0.2f', lambdaMax));
% hold on

%  for each_w_0 = 1: size(W_0 , 1)
    % reset
    iter = 0;
    j = 0;
    stop = 0;
    E = [];
    d = size(W_0, 1); % num of features
    D = zeros(d, C);
    tmp = [];
    
    r = 0.3; % penalty parameter
   
    while ~stop
        % update weights for each class
        for k = 1 : C
            w = W_0 (:, k);
            [d, w] = update_w(X, T(:, k) , S(:, k), w, lambdaMax, W_0, r);
            D (:, k) = d;
            W_0(:,k) = w;
        end
        
        % calculate the L-infinity norm of the CE's gradient for convergance
        inf_norm = norm(D, Inf);
%         tmp = [tmp inf_norm];
        if (inf_norm <= tol)
            stop = 1;
        else
            stop = 0;
        end

         % calculate S w.r.t. updated weights for the next iteration
        S = calculate_S(X, W_0);
        
         E = [E calculate_entropy(T, S, C)]; % norm of E is useful for step length assessment
                                         % if too large, we bounce
                                         % if too small, we are approaching
                                         % to the stationary point
        
%         E = [E regularized_CE(T, S, W_0, r)];
        
        % update lambda (step-halving)
        iter = iter + 1;
        big_i = big_i + 1;
        if (size(E,2) > 1)
            if E (end) >= E (end -1) % THEN CHANGE THE EQUALITY SIGN
                lambdaMax = alpha * lambdaMax;
                j = j + 1;
            end
        end
        
       num_of_iters = [num_of_iters iter];
       
        if iter == maxIter
            break;
        end   
%         if mod(iter,5) == 0
%             plotDecisionBoundary(Xtrain,Ltrain,50,'MNR', W_0);
%         end
    end

    % plot CE w.r.t. number of iterations
    % task 3 a)
%     plot(num_of_iters, E);
%     xlabel('iteration');
%     ylabel ('Cross-Entropy');
%     xlim([1 100]);


end

function [d, w] = update_w (X, T, S, w, lambdaMax, W, r)
% X is a vector here (the 1st, 2nd, or 3rd column of the original Xtrain)
% T is a vector here (the 1st, 2nd, or 3rd column of the original T)

% the goal: min f(X + lambda * d); where X belongs to the previous step
%                                  and d belongs to current step
% calculate cross-entropy (CE) as our loss function (gradient of loss function
% which is the gradient of our cross-entropy w.r.t. W)
d = calculate_gradient_CE(X,S,T); % descent direction
% d = calculate_gradient_regularizedCE(X, S, T, W, r); % descent direction for regularized CE

update = lambdaMax * d; % update: step length * descent direction
w = w - update;

end

