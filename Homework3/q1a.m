clc
clear

%train data
train_x = -1:0.05:1;
train_y = 1.2*sin(pi*train_x) - cos(2.4*pi*train_x) + 0.3*randn(1, size(train_x, 2));
%test data
test_x = -1:0.01:1;
test_y = 1.2*sin(pi*test_x) - cos(2.4*pi*test_x);

%exact interpolation on train
r = train_x'-train_x;
gaussian_rbf = exp ( (r.^2) / (-2*((0.1)^2)));
w = gaussian_rbf \ train_y';

%predict y
r_test = test_x' - train_x;
gaussian_rbf_test = exp ( (r_test.^2) / (-2*((0.1)^2)));
y_predict = (gaussian_rbf_test*w)';
%predict x
r_train = train_x' - train_x;
gaussian_rbf_train = exp ( (r_test.^2) / (-2*((0.1)^2)));


hold on
plot(train_x,train_y,'o')
plot(test_x,y_predict)
plot(test_x,test_y)
legend('Train points','RBFN','Ground Truth', 'Location', 'Best')