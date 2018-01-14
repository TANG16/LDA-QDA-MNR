function [x_l, y_l] = getLabel (i , j)
% GETLABEL calculates what are two x and y labels of the plots given iand j

x_l  = lower('Sepal');
y_l = lower('');

if i == 1
    x_l = lower('Sepal Length');
elseif i == 2
    x_l = lower('Sepal Width');
elseif i == 3
    x_l = lower('Petal Length');
elseif i == 4
    x_l = lower('Petal Width');
end

if j == 1
    y_l = lower('Sepal Length');
elseif j == 2
    y_l = lower('Sepal Width');
elseif j == 3
    y_l = lower('Petal Length');
elseif j == 4
   y_l = lower('Petal Width');
end

end