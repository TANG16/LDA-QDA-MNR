
% function plotDecisionBoundary(data, labels, n, classifier, Weights)
function plotDecisionBoundary(data, labels, n, classifier)
% function plotDecisionBoundary()
%plotDecisionBoundary
% The following function code is borrowed from this website:
% Ref: http://www.peteryu.ca/tutorials/matlab/visualize_decision_boundaries

% training: Structure of two cells each of which contains a matrix of
%           mx2; where m is a number between 0 and n, and 2 represents xy
%           position of the point on the screen
%           cell1 includes data samples from only class1
%           cell2 includes data samples from only class2


% modified by Maryam Najafi on Sep 27, 2016
% Some explanatory sentences above lines are from the author of the 
% reference link.

% Author: Maryam Najafi
% Created Date: Oct 25, 2016

% for LDA
global MU; global C; global P; global S_test
% for MNR
global Weights; 
global lambdaMax
training = prepareDatasetToPlot(data, labels);

% set up the domain over which you want to visualize the decision
% boundary
xrange = [0 10];
yrange = [0 10];
% step size for how finely you want to visualize the decision boundary.
inc = 0.1;
 
% generate grid coordinates. this will be the basis of the decision
% boundary visualization.
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
% size of the (x, y) image, which will also be the size of the 
% decision boundary image that is used as the plot background.
image_size = size(x);
 
xy = [x(:) y(:)]; % make (x,y) pairs as a bunch of row vectors.
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy); % number of (x,y) pairs
 
global unknown_label
idx = [];
X = [];
% make the background unlabeled matrix X
for i=0:inc:10
    for j = 0:inc:10
        X = [X; [i,j]];
    end
end

% classify
if (isequal(classifier, 'LDA') || isequal(classifier, 'QDA'))
    [Ypred, Posteriors] = da_classify(X,MU,C,P);
end

if (isequal(classifier, 'MNR'))
    [Ypred, Posteriors] = mnr_classify(X,Weights);
end

% reshape the idx (which contains the class label) into an image.
decisionmap = reshape(Ypred, image_size);
figure;
 
%show the image
imagesc(xrange,yrange,decisionmap);
hold on;
set(gca,'ydir','normal');
 
% colormap for the classes:
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.9 0.9 1; 0.95 1 0.95];
colormap(cmap);
 
% plot the class training data.
hold on
plot(training{1}(:,1),training{1}(:,2), 'ro');
plot(training{2}(:,1),training{2}(:,2), 'bo');
plot(training{3}(:,1),training{3}(:,2), 'go');

% include legend
% legend('Class 1', 'Class 2','Location','NorthOutside', ...
%     'Orientation', 'horizontal');
legend('Class 1', 'Class 2', 'Class 3', 'Location','NorthOutside', ...
    'Orientation', 'horizontal');

if (isequal(classifier, 'QDA'))
    title (sprintf('IrisReduced dataset classified using QDA - Isotropic classifier'));
elseif (isequal(classifier, 'LDA'))
    title (sprintf('IrisReduced dataset classified using LDA - Isotropic classifier'));
end

if (isequal(classifier, 'MNR'))
    title (sprintf('IrisReduced dataset classified using MNR classifier \n step length = %0.2d', lambdaMax));
end


% label the axes.
xlabel('Sepal Width');
ylabel('Sepal Length');
hold on;
set(gca,'ydir','normal');

end