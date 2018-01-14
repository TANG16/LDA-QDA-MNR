function training = prepareDatasetToPlot(X, L)
% prepareDatesetToPlot

% Author: Maryam Najafi
% Created Date: Sep 27, 2016

training = cell(3,1);
training{1} = X((L(:) == 1), 1:2); % exclude labels
training{2} = X((L(:) == 2), 1:2); % exclude labels
training{3} = X((L(:) == 3), 1:2); % exclude labels

end