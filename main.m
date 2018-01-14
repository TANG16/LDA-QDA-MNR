% Dataset: iris.data.shuffled.mat
% Dataset download link: https://fit.instructure.com/courses/491870/files/37286283/download?wrap=1

% The objective is to classify training samples using LDA/QDA and MNR
% algorithms. QDA: Quadric Discirminant Analysis Classifier
%             LDA: Linear Discriminant Analysis Classifier
%             MNR: Multinomial Regression


% Author: Mary Najafi
% Created date: Oct 23, 2016
% Last modified date: Oct 31, 2016

close all
clc
clear all
% data preparation
load ('iris.data.shuffled.mat');
pattern = Pattern;
clear Pattern
label = Label;
clear Label

N_iris = length(label);

%% task 1. a) Plot scatter plots for every pair over 4 features
% features are Septal Length, Sepal Width, Petal Length, Petal Width

for i = 1 : 4
    if i ~= 4
        for j = i + 1: 4
%             figure();
            x = getcolumn(pattern(1:N_iris, 1:4),i); 
            y = getcolumn(pattern(1:N_iris,1:4),j);
%             scatter (x(label == 1), y(label == 1), 'r'); hold on; scatter (x(label == 2), y(label == 2), 'b'); hold on;  scatter (x(label == 3), y(label == 3), 'g'); 
%             xlim([0 10]); ylim([0 10]); title(sprintf('iris distribution for columns %d and %d', i, j));
%             [x_label y_label] = getLabel(i,j);
%             xlabel(sprintf('%s',x_label)); ylabel(sprintf('%s',y_label));
        end
    end
end

% Form IrisReduced dataset containing only column 3 and 4
% feature #3 is Petal length and feature #4 is Petal width

irisReduced = getcolumn(pattern (1:N_iris, 3:4), 1:2); % 1:2 correspond to returned columns 3:4

%% Form a training set out of first 50 samples of irisReduced
% prepare data for train and test
N = 50; N_tst = N_iris - N;
global S_label;
global S_label_tst
global MU; global C; global P; global S_test
S_train = getcolumn(irisReduced(1:N, 1:2), 1:2);
% S_test = getcolumn (irisReduced(N+1:size(irisReduced, 1), 1:2), 1:2); 
S_label = label(1:N);
% S_label_tst = label(N+1: length(label));

%% task 1. b) LDA
% 
% % 2_1. General-Case LDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_traintmp, S_labeltmp, 1);
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over General-Case LDA = %d ' , err/N)
% plotDecisionBoundary(S_train, S_label,N,'LDA');

% % 2_2 Naiive-Bayes LDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_train, S_label, 2);
%     
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over Naive-Bayes LDA = %d', err/N)
% plotDecisionBoundary(S_train, S_label,N,'LDA');
% 
% % 2_3 Isotropic LDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_train, S_label, 3);
%     
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over Isotropic LDA = %d' , err/N)
% plotDecisionBoundary(S_train, S_label,N,'LDA');
% 
% %% task 1. c) QDA
% 
% % 3_1. General-Case QDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_traintmp, S_labeltmp, 4);
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over General-Case QDA = %d ' , err/N)
% 
% 
% % 3_2 Naiive-Bayes QDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_traintmp, S_labeltmp, 5);
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over Naive-Bayes QDA = %d ' , err/N)
% 
% % 3_3 Isotropic QDA
% err = 0;
% for i = 1 : N
%     S_test(1, :) = S_train(i, : );
%     S_traintmp = S_train;
%     S_traintmp(i, : ) = [];
%     S_label_tst = S_label(i);
%     S_labeltmp = S_label;
%     S_labeltmp(i) = [];
%     
%     % train
%     [MU, C, P] = da_train(S_traintmp, S_labeltmp, 6);
%     % Test using Leave-One-Out Cross-Validation approach
%     [pred_labels, Posteriors] = da_classify(S_test,MU,C,P);
%     
%     % validate
%     if (pred_labels ~= S_label(i))
%         err = err + 1;
%     end
% end
% sprintf('LOOCV over Isotropic QDA = %d ' , err/N)

%% task 1. d) Decision Boundary
% plotDecisionBoundary(S_train, S_label,50,'QDA');

%% task 3. a) MNR classification
% Weights = [5.46287169635868,0.957894880723720,-4.21110520899040;-0.472172972206489,0.951300438760785,0.819723662445361;-2.06994511154717,-0.407540623241836,3.32460347098023;];
global lambdaMax
maxIter = 100; % max possible backtracking iterations
lambdaMax = 0.1; % step length e.g. 0.2 and 0.1
alpha = .6; % step length gets affected by multiplying it by alpha
tol = 2; % termination condition
Ltrain = S_label;
Xtrain = S_train;
global Weights


% LOOCV
err = 0;
for i = 1 : N
    Xtest(1, :) = Xtrain(i, : );
    X_traintmp = Xtrain;
    X_traintmp(i, : ) = [];
    L_label_tst = Ltrain(i);
    L_labeltmp = Ltrain;
    L_labeltmp(i) = [];

[Weights, CEvalues, status] = mnr_train(X_traintmp, L_labeltmp, maxIter, lambdaMax, alpha, tol);
% The cross-entropy and Regularized cross-entropy plots are in mnr_train.

% now after getting the optimal weights (from training), classify.
[Lpred, scores] = mnr_classify(Xtest, Weights);

% T = create_t(Xtest, L_label_tst);
% calculate error
if (Lpred ~= L_label_tst)
    err = err + 1;
end

end

avgErr = err / N;
sprintf('LOOCV over MNR error = %0.2d%%' , avgErr*100)

%% task 3. b) decision boundary
plotDecisionBoundary(Xtrain,Ltrain,N,'MNR');

%% task 4. b) Regularized Logistic Regression in Multinomial Regression
% add penalty term to the Cross-Entropy


