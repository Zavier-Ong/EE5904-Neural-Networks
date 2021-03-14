clc
clear
close all

% Matric A0138993L
% Classes chosen: 9 and 3
load('characters10.mat');

%imshow(reshape(train_data(2997,:), [28,28]));
train_idx = find(train_label == 3 | train_label == 9);
% 9(K) --> 1 and 3(R) --> 0
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

rng(3);
k = 2;
center_idx = randperm(size(train_x, 1));
curr_cen = train_x(center_idx(1:k), :);
old_cen = zeros(size(curr_cen));

%K means clustering
while ~isequal(curr_cen, old_cen)
    old_cen = curr_cen;
    % assignment
    distance = pdist2(old_cen, train_x);
    [~, label] = min(distance, [], 1);
    % updating
    curr_cen(1,:) = mean(train_x(label==1, :), 1);
    curr_cen(2,:) = mean(train_x(label==2, :), 1);
end

%obtained centers
fig = figure();
sgtitle('Obtained centers');
subplot(121);
imshow(reshape(curr_cen(1,:), [28,28]));
subplot(122);
imshow(reshape(curr_cen(2,:), [28,28]));
saveas(fig, 'q2c_k_centers.png');
%mean of training image
fig = figure();
sgtitle('Mean of training images');
subplot(121);
imshow(reshape(mean(train_x(TrLabel==1, :), 1), [28,28]));
subplot(122);
imshow(reshape(mean(train_x(TrLabel==0, :), 1), [28,28]));
saveas(fig, 'q2c_mean_imgs.png');

%training
r_factors = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000];
sigma = 100;
for i=r_factors
    lambda = i;
    r_train = pdist2(train_x, curr_cen, 'squaredeuclidean');
    phi = exp( r_train / (-2*((sigma)^2)) );
    w = ((phi'*phi) + lambda*eye(size(phi, 2))) \ (phi'*TrLabel);
    %TrPred
    TrPred = (phi*w)';

    %TePred
    r_test = pdist2(test_x, curr_cen, 'squaredeuclidean');
    %gaussian rbf
    phi = exp( r_test / (-2*((sigma)^2)) );
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
    title(sprintf('K-Means Clustering (lambda=%.4f)', i))
    saveas(fig,sprintf('q2c_lambda_%.4f.png',i))
    hold off
end
