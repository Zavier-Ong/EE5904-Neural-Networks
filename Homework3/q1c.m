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

r_factors = [0, 0.001, 0.01, 0.1, 1, 10, 100]
for i = r_factors
    % exact interpolation on train
    % no need to square or sqrt since x has only 1 value
    lambda = i;
    r = abs(train_x'-train_x);
    % gaussian rbf
    phi = exp( (r.^2) / (-2*((0.1)^2)) );
    % applying regularization method to determine new weights
    w = pinv((phi'*phi) + lambda*eye(size(phi, 2))) * (phi'*train_y');

    %predict y
    r_test = abs(test_x' - train_x);
    phi = exp( (r_test.^2) / (-2*((0.1)^2)) );
    y_predict = (phi*w)';

    fig = figure();
    hold on
    plot(train_x,train_y,'ok')
    plot(test_x,y_predict, 'r')
    plot(test_x,test_y, 'b')
    legend('Train points (with noise)','RBFN','Expected', 'Location', 'Best')
    title(sprintf('Exact Interpolation (lambda=%.3f)', i))
    saveas(fig, sprintf('q1c_%.3f.png', i))
    hold off

    %MSE of test
    mse_test = sum((y_predict-test_y).^2)/size(y_predict, 2);
    fprintf('Mean Square Error on test (lambda = %0.3f): %f\n', i, mse_test);
end
