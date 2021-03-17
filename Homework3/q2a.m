clc
clear
close all

% Matric A0138993L
% Classes chosen: 9 and 3
load('characters10.mat');

train_idx = find(train_label == 3 | train_label == 9);
% 9 --> 1 and 3 --> 0
TrLabel = train_label(train_idx);
TrLabel(TrLabel == 9) = 1;
TrLabel(TrLabel == 3) = 0;

train_x = train_data(train_idx, :);
% normalizing train data
train_x = mat2gray(train_x(:,:));

test_idx = find(test_label == 3 | test_label == 9);
TeLabel = test_label(test_idx);
TeLabel(TeLabel == 9) = 1;
TeLabel(TeLabel == 3) = 0;

test_x = test_data(test_idx, :);
% normalizing test data
test_x = mat2gray(test_x(:,:));

r_factors = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100];
sd= 100;
for i = r_factors
    % exact interpolation on train
    lambda = i;
    r_train = pdist2(train_x, train_x, 'squaredeuclidean');
    % gaussian rbf
    phi = exp( r_train / (-2*((sd)^2)) );
    % applying regularization method to determine new weights
    if (i == 0)
        w = phi \ TrLabel;
    else
        w = ((phi'*phi) + lambda*eye(size(phi, 2))) \ (phi'*TrLabel);
    end
    
    % TrPred
    TrPred = (phi*w)';
    
    %TePred
    r_test = pdist2(test_x, train_x, 'squaredeuclidean');
    %gaussian rbf
    phi = exp( r_test / (-2*((sd)^2)) );
    TePred = (phi*w)';

    fig = figure();
    TrAcc = zeros(1,1000);
    TeAcc = zeros(1,1000);
    thr = zeros(1,1000);
    TrN = length(TrLabel);
    TeN = length(TeLabel);
    for j = 1:1000
        t = (max(TrPred)-min(TrPred)) * (j-1)/1000 + min(TrPred);
        thr(j) = t;
        TrAcc(j) = (sum(TrLabel(TrPred<t)==0) + sum(TrLabel(TrPred>=t)==1)) / TrN;
        TeAcc(j) = (sum(TeLabel(TePred<t)==0) + sum(TeLabel(TePred>=t)==1)) / TeN;
    end
    hold on
    plot(thr, TrAcc, '-^r');
    plot(thr, TeAcc, '-xb');
    legend('Train','Test');
    title(sprintf('Exact Interpolation (lambda=%.4f)', i))
    saveas(fig,sprintf('q2a_lambda_%.4f.png',i))
    hold off
end
