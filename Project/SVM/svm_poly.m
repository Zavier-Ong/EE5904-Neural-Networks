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

%Hard margin C = 10e6
C = 10e6;
p = [2, 3, 4, 5];

%checking for kernel suitability
for i=1:size(p, 2)
    K = (strd_train'*strd_train+1).^p(i);
    svm_helper.check_mercer(K);
    %Task (Polynomial kernel)

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
    adk_term = alpha.*train_label.*K; 
    adK = sum(adk_term,1)';
    boi = train_label(idx) - adK(idx);
    bo = mean(boi);

    %Task 2 (Test set)
    train_acc = svm_helper.get_poly_kernel_acc(alpha, bo, p(i), strd_train, train_label, strd_train, train_label);
    test_acc = svm_helper.get_poly_kernel_acc(alpha, bo, p(i), strd_train, train_label, strd_test, test_label);
    fprintf('Train accuracy of hard-margin SVM with poly kernel (p=%d): %.2f%%\n', p(i), train_acc*100);
    fprintf('Test accuracy of hard-margin SVM with poly kernel (p=%d): %.2f%%\n', p(i), test_acc*100);

end



