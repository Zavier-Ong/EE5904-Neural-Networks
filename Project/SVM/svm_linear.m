clc
clear

load('train.mat')
load('test.mat')

%preprocessing
%find mean of each row
mu = mean(train_data, 2);
sd = std(train_data, 0, 2);
%standardization method
strd_train = (train_data-mu)./sd;
strd_test = (test_data-mu)./sd;

%checking for kernel suitability
K = strd_train'*strd_train;
svm_helper.check_mercer(K);

%Hard margin C = 10e6
C = 10e6;

%Task 1 (Linear kernel)
H = K.*(train_label*train_label');
f = -ones(2000, 1);
A = [];
b= [];
Aeq = train_label';
beq = 0;
lb = zeros(2000, 1);
ub = ones(2000, 1)*C;
x0 = [];
options = optimset('LargeScale','off','MaxIter', 1000);
%calc alpha - quadprog
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);
idx = find(alpha>1e-4);
%discriminant function g(x)
wo = sum(alpha'.*train_label'.*strd_train, 2);
boi = 1./train_label(idx) - strd_train(:, idx)'*wo;
bo = mean(boi);

%Task 2 (Test set)
train_acc = svm_helper.get_linear_kernel_acc(wo, bo, strd_train, train_label);
test_acc = svm_helper.get_linear_kernel_acc(wo, bo, strd_test, test_label);
fprintf('Train accuracy of hard-margin SVM with linear kernel: %.2f%%\n', train_acc*100);
fprintf('Test accuracy of hard-margin SVM with linear kernel: %.2f%%\n', test_acc*100);

