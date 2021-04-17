clc
clear

load('train.mat')
load('test.mat')
load('eval.mat')

%preprocessing
%find mean of each row
mu = mean(train_data, 2);
sd = std(train_data, 0, 2);
%standardization method
strd_train = (train_data-mu)./sd;
strd_test = (test_data-mu)./sd;
strd_eval = (eval_data-mu)./sd;

n_feat = size(strd_train, 1);

%Testigng different parameters
%gamma = [0.001, 0.01, 0.1, 1, 10];
%C = [0.001, 0.01, 0.1, 1, 10, 100];
%Finalized results
gamma = (0.01);
C = (10);

acc_list = zeros((size(C, 2)*size(gamma, 2)), 6);
eval_pred_list = zeros((size(C,2)*size(gamma,2)), 600);
list_idx = 1;

for i = 1:size(C, 2)
    for j = 1:size(gamma, 2)
        %checking for kernel suitability
        %gram matrix (K) = exp(-gamma*squared euclidean distance .|.
        %feature vectors
        K = exp(-gamma(j)*pdist2(strd_train', strd_train', 'squaredeuclidean'));
        isSuitable = svm_helper.check_mercer(K);
        
        H = K.*(train_label*train_label');
        f = -ones(2000, 1);
        A = [];
        b= [];
        Aeq = train_label';
        beq = 0;
        lb = zeros(2000, 1);
        ub = ones(2000, 1)*C(i);
        x0 = [];
        options = optimset('LargeScale','off','MaxIter', 1000);
        %calc alpha - quadprog
        alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, x0, options);
        idx = find(alpha>1e-4 & alpha<=C(i));
        %discriminant function g(x)
        adk_term = alpha.*train_label.*K; 
        adK = sum(adk_term)';
        boi = train_label(idx) - adK(idx);
        bo = mean(boi);

        %Task 2 (Test set)
        [train_acc, train_predicted] = svm_helper.get_rbf_kernel_acc(alpha, bo, gamma(j), strd_train, train_label, strd_train, train_label);
        [test_acc, test_predicted] = svm_helper.get_rbf_kernel_acc(alpha, bo, gamma(j), strd_train, train_label, strd_test, test_label);
        [eval_acc, eval_predicted] = svm_helper.get_rbf_kernel_acc(alpha, bo, gamma(j), strd_train, train_label, strd_eval, eval_label);
        acc_list(list_idx,:) = [isSuitable gamma(j) C(i) train_acc test_acc, eval_acc];
        eval_pred_list(list_idx, :) = eval_predicted;
        list_idx = list_idx + 1;
    end
end

for i=1:size(acc_list, 1)
    isSuitable = acc_list(i, 1);
    gamma = acc_list(i, 2);
    c = acc_list(i, 3);
    train_acc = acc_list(i, 4);
    test_acc = acc_list(i, 5);
    eval_acc = acc_list(i, 6);
    if isSuitable
        fprintf('Kernel candidate is admissible\n');
    else
        fprintf('Kernel candidate is not admissible\n');
    end
    fprintf('Train accuracy of soft-margin SVM with RBF kernel (c=%.3f, gamma=%.3f): %.2f%%\n', c, gamma, train_acc);
    fprintf('Test accuracy of soft-margin SVM with RBF kernel (c=%.3f, gamma=%.3f): %.2f%%\n', c, gamma, test_acc);
    fprintf('Eval accuracy of soft-margin SVM with RBF kernel (c=%.3f, gamma=%.3f): %.2f%%\n\n', c, gamma, eval_acc);
end