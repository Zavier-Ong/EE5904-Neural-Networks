clc
clear
close all

%for reproducibility
rng(3);
% train data
train_x = -1:0.05:1;
train_y = 1.2*sin(pi*train_x) - cos(2.4*pi*train_x) + 0.3*randn(1, size(train_x, 2));
% test data
test_x = -1:0.01:1;
test_y = 1.2*sin(pi*test_x) - cos(2.4*pi*test_x);

% randomly select 20 centers
m = 20;
center_idx = randperm(size(train_x, 2));
mew_x = train_x(center_idx(1:m));
mew_y = train_y(center_idx(1:m));

% no need to square or sqrt since x has only 1 value
r = abs(train_x'-mew_x);
dist_cen = abs(mew_x'-mew_x);
% maximum dist between chosen centers
dmax = max(dist_cen, [], 'all');
% rbf
phi = exp( -(m/dmax^2) * r.^2 );
w = phi \ train_y';

%predict y
r_test = abs(test_x' - mew_x);
% rbf
phi = exp( -(m/dmax^2) * r_test.^2 );
y_predict = (phi*w)';

fig = figure();
hold on
plot(mew_x,mew_y,'ok')
plot(test_x,y_predict, 'r')
plot(test_x,test_y, 'b')
legend('Train points (with noise)','RBFN','Expected', 'Location', 'Best')
title('Fixed Centres Selected at Random')
saveas(fig, 'q1b.png')
hold off

%MSE of test
mse_test = sum((y_predict-test_y).^2)/size(y_predict, 2);
fprintf('Mean Square Error on test: %f\n', mse_test);

