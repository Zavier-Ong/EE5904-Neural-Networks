clc
clear
close all

% Matric A0138993L
% Classes chosen: 9 and 3
load('characters10.mat');

%imshow(reshape(train_data(2997,:), [28,28]));
train_idx = find(train_label == 3 | train_label == 9);
% 9 --> 1 and 3 --> 0
TrLabel = train_label(train_idx);
TrLabel(TrLabel == 9) = 1;
TrLabel(TrLabel == 3) = 0;

train_x = train_data(train_idx, :);
% casting to double to remove warning when normalizing data
train_x = mat2gray(train_x(:,:));

test_idx = find(test_label == 3 | test_label == 9);
TeLabel = test_label(test_idx);
TeLabel(TeLabel == 9) = 1;
TeLabel(TeLabel == 3) = 0;

test_x = test_data(test_idx, :);
% casting to double to remove warning when normalizing data
test_x = mat2gray(test_x(:,:));

% randomly select 100 centers
m = 100;
% seed for reproducibility
rng(3)
center_idx = randperm(size(train_x, 1));
selected_train_x = train_x(center_idx(1:m), :);
selected_TrLabel = TrLabel(center_idx(1:m), :);

r_train = pdist2(train_x, selected_train_x, 'squaredeuclidean');
dist_cen = pdist2(selected_train_x, selected_train_x, 'squaredeuclidean');
dmax_squared = max(dist_cen, [], 'all');
sigma = sqrt(dmax_squared/(2*m));

vary_width = [sigma, 0.1, 1, 10, 100, 1000, 10000];
for i = vary_width
    width = i;
    phi = exp( -(r_train / (2*(width^2))) );
    w = phi \ TrLabel;
    
    %TrPred
    TrPred = (phi*w);
    
    %TePred
    r_test = pdist2(test_x, selected_train_x, 'squaredeuclidean');
    phi = exp ( -(r_test / (2*(width^2))) );
    TePred = (phi*w);
    
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
    title(sprintf('Fixed Centers Selected at Random (lambda=%.3f)', i))
    saveas(fig,sprintf('q2b_lambda_%.3f.png',i))
    hold off
end
    
