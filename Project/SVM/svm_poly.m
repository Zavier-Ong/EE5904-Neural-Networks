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
C = [10e6, 0.1, 0.6, 1.1, 2.1];
p = [1, 2, 3, 4, 5];

%checking for kernel suitability
acc_list = zeros((size(C, 2)*size(p, 2)-1), 5);
list_idx = 1;
for i = 1:size(C, 2)
    for j=1:size(p, 2)
        if C(i)==10e6 && p(j)==1
            continue;
        end
        K = (strd_train'*strd_train+1).^p(j);
        isSuitable = svm_helper.check_mercer(K);
        %Task (Polynomial kernel)

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
        idx = find(alpha>1e-4);
        %discriminant function g(x)
        adk_term = alpha.*train_label.*K; 
        adK = sum(adk_term,1)';
        boi = train_label(idx) - adK(idx);
        bo = mean(boi);

        %Task 2 (Test set)
        train_acc = svm_helper.get_poly_kernel_acc(alpha, bo, p(j), strd_train, train_label, strd_train, train_label);
        test_acc = svm_helper.get_poly_kernel_acc(alpha, bo, p(j), strd_train, train_label, strd_test, test_label);
        acc_list(list_idx,:) = [isSuitable p(j) C(i) train_acc test_acc];
        list_idx = list_idx + 1;
    end
end

for i=1:size(acc_list, 1)
    isSuitable = acc_list(i, 1);
    p = acc_list(i, 2);
    c = acc_list(i, 3);
    train_acc = acc_list(i, 4);
    test_acc = acc_list(i, 5);
    if isSuitable
        fprintf('Kernel candidate is admissible\n');
    else
        fprintf('Kernel candidate is not admissible\n');
    end
    fprintf('Train accuracy of hard-margin SVM with poly kernel (c=%.1f, p=%d): %.2f%%\n', c, p, train_acc);
    fprintf('Test accuracy of hard-margin SVM with poly kernel (c=%.1f, p=%d): %.2f%%\n\n', c, p, test_acc);
end

