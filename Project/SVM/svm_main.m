clc
clear

load('train.mat')
load('test.mat')

%preprocessing
%find mean of each row
mew = mean(train_data, 2);
sd = std(train_data, 1, 2);
%standardization method
stand_train = (train_data-mew)./sd;
stand_test = (test_data-mew)./sd;

%checking got kernel suitability
gram_matrix = stand_train'*stand_train;
check_mercer(gram_matrix);

%Hard margin C = 10e6
C = 10e6;

%Task 1 (Linear kernel)
H = gram_matrix.*(train_label*train_label');
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
wo = sum(alpha'.*train_label'.*stand_train, 2);
bo = mean(1./train_label(idx) - stand_train(:, idx)'*wo);

%Task 2 (Test set)
train_acc = get_acc(wo, bo, stand_train, train_label);
test_acc = get_acc(wo, bo, stand_test, test_label);
fprintf('Train accuracy of hard-margin SVM with linear kernel: %.2f%%\n', train_acc*100);
fprintf('Test accuracy of hard-margin SVM with linear kernel: %.2f%%\n', test_acc*100);

function check_mercer(matrix)
    e = eig(matrix);
    if min(e) < -1e-4
        fprintf('Kernel candidate is not admissible\n');
        return;
    else
        fprintf('Kernel candidate is admissible\n');
        return;
    end

end

function acc = get_acc(wo, bo, data, label)
    g = wo'*data+bo;
    pred_label = sign(g)';
    acc = mean(pred_label==label, 'all')
end

        