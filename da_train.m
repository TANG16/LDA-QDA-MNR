function [Means, Covariances, Priors] = da_train(Xtrain, Ltrain, classifier_type)
%DA_TRAIN Train the dataset using LDA/QDA classifiers

%   input arguments
%   Xtrain: a matrix of size N_train by D, containing training samples
%   Ltrain: a column vec of size N_train, containing corresponding labels
%   classifier_type: 1, 2, 3 for LDA general, Naiive Bayes, and Isotropic
%                    4, 5, 6 for QDA general, Naiive Bayes, and Isotropic

%   output params:
%   Means: a matrix of size k by D; where k is number of classes. The row k
%          corresponds to the estimated mean of kth class
%   Covariances: a matrix of size DxDxC that contains DxD estimated 
%                covariance matrices for each class C
%   Priors: a vec of size k containing each one of classes' estimate priors

% We assume that Xtrain contains at least one sample of each class C= 1,2,3

N = size(Xtrain,1);
switch (classifier_type)
    case 1
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];

        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
        % class 1: Setosa
        X = Xtrain((Ltrain==1), 1 : 2);
        S_1 = ( X - ones(size(X, 1), 1) * mu_1)' * ( X - ones(size(X, 1), 1) * mu_1)/ size(X,1);
        % class 2: Versicolor
        X = Xtrain((Ltrain==2), 1 : 2);
        S_2 = ( X - ones(size(X, 1), 1) * mu_2)' * ( X - ones(size(X, 1), 1) * mu_2)/ size(X,1);
        % class 3: Virginica
        X = Xtrain((Ltrain==3), 1 : 2);
        S_3 = ( X - ones(size(X, 1), 1) * mu_3)' * ( X - ones(size(X, 1), 1) * mu_3)/ size(X,1);
        % Pooled Sample Covariance Matrix
        C = p1 * S_1 + p2 * S_2 + p3 * S_3;
        Covariances(:,:,1) = C; Covariances(:,:,2) = C; Covariances(:,:,3) = C;
        
    case 2
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];
        
        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
        % class 1: Setosa
        X_f1 = Xtrain((Ltrain==1), 1); % f1 is feature (column) 1
        X_f2 = Xtrain((Ltrain==1), 2);
%         C_1_f1 = sum(sum((X_f1 - ones(size(X_f1, 1), 1) * mu_1(1))' * (X_f1 - ones(size(X_f1, 1), 1) * mu_1(1))));
%         C_1_f2 = sum(sum((X_f2 - ones(size(X_f2, 1), 1) * mu_1(2))' * (X_f2 - ones(size(X_f2, 1), 1) * mu_1(2))));
        C_1_f1 = var(X_f1);
        C_1_f2 = var(X_f2);

        % class 2: Versicolor
        X_f1 = Xtrain((Ltrain==2), 1);
        X_f2 = Xtrain((Ltrain==2), 2);
%         C_2_f1 = sum(sum((X_f1 - ones(size(X_f1, 1), 1) * mu_2(1))' * (X_f1 - ones(size(X_f1, 1), 1) * mu_2(1))));
%         C_2_f2 = sum(sum((X_f2 - ones(size(X_f2, 1), 1) * mu_2(2))' * (X_f2 - ones(size(X_f2, 1), 1) * mu_2(2))));
        C_2_f1 = var(X_f1);
        C_2_f2 = var(X_f2);

        % class 3: Virginica
        X_f1 = Xtrain((Ltrain==3), 1);
        X_f2 = Xtrain((Ltrain==3), 2);
%         C_3_f1 = sum(sum((X_f1 - ones(size(X_f1, 1), 1) * mu_3(1))' * (X_f1 - ones(size(X_f1, 1), 1) * mu_3(1))));
%         C_3_f2 = sum(sum((X_f2 - ones(size(X_f2, 1), 1) * mu_3(2))' * (X_f2 - ones(size(X_f2, 1), 1) * mu_3(2))));
        C_3_f1 = var(X_f1);
        C_3_f2 = var(X_f2);
        
        % 
        v_1 = (C_1_f1 * p1 + C_2_f1 * p2 + C_3_f1 * p3); % v_d; d = 1
        v_2 = (C_1_f2 * p1 + C_2_f2 * p2 + C_3_f2 * p3); % v_d; d = 2
        
        % C = diag(v)
        Covariances(:,:,1) = diag([v_1 v_2]);
        Covariances(:,:,2) = diag([v_1 v_2]);
        Covariances(:,:,3) = diag([v_1 v_2]);
        
    case 3
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];
        
        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
%         % class 1: Setosa
%         X = Xtrain((Ltrain==1), 1 : 2);
%         C_1 = sum(sum((X - ones(size(X, 1), 1) * mu_1)' * (X - ones(size(X, 1), 1) * mu_1)));
%         % class 2: Versicolor
%         X = Xtrain((Ltrain==2), 1 : 2);
%         C_2 = sum(sum((X - ones(size(X, 1), 1) * mu_2)' * (X - ones(size(X, 1), 1) * mu_2)));
%         % class 3: Virginica
%         X = Xtrain((Ltrain==3), 1 : 2);
%         C_3 = sum(sum((X - ones(size(X, 1), 1) * mu_3)' * (X - ones(size(X, 1), 1) * mu_3)));
%         % 
%         v = ( 1 / N) * (C_1 + C_2 + C_3);


        X_f1 = Xtrain((Ltrain==1), 1); % f1 is feature (column) 1
        X_f2 = Xtrain((Ltrain==1), 2);
        C_1_f1 = var(X_f1);
        C_1_f2 = var(X_f2);

        % class 2: Versicolor
        X_f1 = Xtrain((Ltrain==2), 1);
        X_f2 = Xtrain((Ltrain==2), 2);
        C_2_f1 = var(X_f1);
        C_2_f2 = var(X_f2);

        % class 3: Virginica
        X_f1 = Xtrain((Ltrain==3), 1);
        X_f2 = Xtrain((Ltrain==3), 2);
        C_3_f1 = var(X_f1);
        C_3_f2 = var(X_f2);
        
        % calculate pooled variance
        v_1 = (C_1_f1 * p1 + C_2_f1 * p2 + C_3_f1 * p3); % v_d; d = 1
        v_2 = (C_1_f2 * p1 + C_2_f2 * p2 + C_3_f2 * p3); % v_d; d = 2
        
        % calculate average over dimensions (features)
        v = (v_1 + v_2) / 2 ; % where 2 is the number of features
        
        Covariances(:,:,1) =  v * eye(2,2);
        Covariances(:,:,2) =  v * eye(2,2);
        Covariances(:,:,3) =  v * eye(2,2);
        
    case 4
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];

        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
        % class 1: Setosa
        X = Xtrain((Ltrain==1), 1 : 2);
        S_1 = ( X - ones(size(X, 1), 1) * mu_1)' * ( X - ones(size(X, 1), 1) * mu_1) / size(X,1);
        % class 2: Versicolor
        X = Xtrain((Ltrain==2), 1 : 2);
        S_2 = ( X - ones(size(X, 1), 1) * mu_2)' * ( X - ones(size(X, 1), 1) * mu_2)/ size(X,1);
        % class 3: Virginica
        X = Xtrain((Ltrain==3), 1 : 2);
        S_3 = ( X - ones(size(X, 1), 1) * mu_3)' * ( X - ones(size(X, 1), 1) * mu_3)/ size(X,1);
        
        Covariances(:,:,1) = S_1; Covariances(:,:,2) = S_2; Covariances(:,:,3) = S_3;
        
    case 5
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];

        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
        % class 1: Setosa
        X_f1 = Xtrain((Ltrain==1), 1); % f1 is feature (column) 1
        X_f2 = Xtrain((Ltrain==1), 2);
        C_1_f1 = var(X_f1);
        C_1_f2 = var(X_f2);
        % class 2: Versicolor
        X_f1 = Xtrain((Ltrain==2), 1);
        X_f2 = Xtrain((Ltrain==2), 2);
        C_2_f1 = var(X_f1);
        C_2_f2 = var(X_f2);
        % class 3: Virginica
        X_f1 = Xtrain((Ltrain==3), 1);
        X_f2 = Xtrain((Ltrain==3), 2);
        C_3_f1 = var(X_f1);
        C_3_f2 = var(X_f2);
        
        v1_1 = C_1_f1; % variance for the dimension 'd' and class 1
        v2_1 = C_1_f2;
        
        v1_2 = C_2_f1;
        v2_2 = C_2_f2;
        
        v1_3 = C_3_f1;
        v2_3 = C_3_f2;
        
        
        % C = diag(v)
        Covariances(:,:,1) = diag([v1_1 v2_1]);
        Covariances(:,:,2) = diag([v1_2 v2_2]);
        Covariances(:,:,3) = diag([v1_3 v2_3]);
        
    case 6
        % 1. calculate prior assumption for all 3 classes >> p_k = N_k/N
        % class 1: Setosa
        N_1 = getN_k(1); p1 = N_1 / N;
        % class 2: Versicolor
        N_2 = getN_k(2); p2 = N_2 / N;
        % class 3: Virginica
        N_3 = getN_k(3); p3 = N_3 / N;
        Priors = [p1; p2; p3];

        % 2. calculate mean for all 3 classes
        % class 1: Setosa
        mu_1 = sum(Xtrain((Ltrain == 1) , 1:2))/ N_1;
        % class 2: Versicolor
        mu_2 = sum(Xtrain((Ltrain == 2) , 1:2))/ N_2;
        % class 3: Virginica
        mu_3 = sum(Xtrain((Ltrain == 3) , 1:2))/ N_3;
        Means = [mu_1; mu_2; mu_3];
        
        % 3. calculate covariance matrices for all 3 classes
        %    in LDA classes share the same covariance matrix
        % class 1: Setosa
        X_f1 = Xtrain((Ltrain==1), 1); % f1 is feature (column) 1
        X_f2 = Xtrain((Ltrain==1), 2);
        C_1_f1 = var(X_f1);
        C_1_f2 = var(X_f2);
        % class 2: Versicolor
        X_f1 = Xtrain((Ltrain==2), 1);
        X_f2 = Xtrain((Ltrain==2), 2);
        C_2_f1 = var(X_f1);
        C_2_f2 = var(X_f2);
        % class 3: Virginica
        X_f1 = Xtrain((Ltrain==3), 1);
        X_f2 = Xtrain((Ltrain==3), 2);
        C_3_f1 = var(X_f1);
        C_3_f2 = var(X_f2);
        
        v1_1 = C_1_f1; % variance for the dimension 'd' and class 1
        v2_1 = C_1_f2;
        
        v1_2 = C_2_f1;
        v2_2 = C_2_f2;
        
        v1_3 = C_3_f1;
        v2_3 = C_3_f2;  
        
        % calculate average over dimensions (features) for 3 classes
        v_1 = (v1_1 + v2_1) / 2;
        v_2 = (v1_2 + v2_2) / 2;
        v_3 = (v1_3 + v2_3) / 2;
        
        Covariances(:,:,1) = v_1 * eye(2,2);
        Covariances(:,:,2) = v_2 * eye(2,2);
        Covariances(:,:,3) = v_3 * eye(2,2);
        
    otherwise
        disp ('error: an incorrect input for the classifier type!');
end

end

