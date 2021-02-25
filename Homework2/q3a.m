clc
clear
close all

%A0138993L
group_id = mod(93, 3)+1;
train_path = 'Face Database\TrainImages\';
test_path = 'Face Database\TestImages\';
x_ext = '*.jpg';
y_ext = '*.att';

train_x_jpg = dir(sprintf('%s%s', train_path, x_ext));
test_x_jpg = dir(sprintf('%s%s', test_path, x_ext));

train_y_att = dir(sprintf('%s%s', train_path, y_ext));
test_y_att = dir(sprintf('%s%s', test_path, y_ext));

%load img into grayscale
train_x = zeros(10201, length(train_x_jpg));
test_x = zeros(10201, length(test_x_jpg));

for i=1:length(train_x_jpg)
    img = imread(sprintf('%s%s', train_path, train_x_jpg(i).name));
    img_size = size(img);
    if (img_size(1) ~= 101 || img_size(2) ~= 101)
        img = imresize(img, [101, 101]);
    end
    grayscale = rgb2gray(img);
    train_x(:, i) = grayscale(:); 
end
for i=1:length(test_x_jpg)
    img = imread(sprintf('%s%s', test_path, test_x_jpg(i).name));
    img_size = size(img);
    if (img_size(1) ~= 101 || img_size(2) ~= 101)
        img = imresize(img, [101, 101]);
    end
    grayscale = rgb2gray(img);
    test_x(:, i) = grayscale(:); 
end

%extracting ground-truth labels
train_y = zeros(length(train_y_att),1);
test_y = zeros(length(test_y_att), 1);

for i=1:length(train_y)
    L = load(sprintf('%s%s', train_path, train_y_att(i).name));
    train_y(i,:) = L(group_id);
end
for i=1:length(test_y)
    L = load(sprintf('%s%s', test_path, test_y_att(i).name));
    test_y(i,:) = L(group_id);
end

%%%%%%%%%%%%%%%%%%%   3a   %%%%%%%%%%%%%%%%%%%
% fig_test = figure();
% h_test = histogram(test_y);
% saveas(fig_test, 'test_label_distribution.png');
% fig_train = figure();
% h_train = histogram(train_y);
% saveas(fig_train, 'train_label_distribution.png');

train_y = train_y.';
test_y = test_y.';
%%%%%%%%%%%%%%%%%%%   3b   %%%%%%%%%%%%%%%%%%%
% 
% net = perceptron;
% %net.performFcn = 'mse';
% [net, tr] = train(net, train_x, train_y);
% 
% %training accuracy
% train_y_pred = net(train_x);
% train_acc = mean(train_y_pred == train_y);
% %test accuracy
% test_y_pred = net(test_x);
% test_acc = mean(test_y_pred == test_y);
% 
% fprintf("Q3b. Rosenblatt's Perceptron\ntraining accuracy: %f, validation accuracy: %f\n", ...
%     train_acc, test_acc);

%%%%%%%%%%%%%%%%%%%   3c   %%%%%%%%%%%%%%%%%%%
% [coeff, score, ~, ~, ~, mu] = pca(train_x.');
% 
% net = perceptron;
% net.performFcn = 'mse';
% [net, tr] = train(net, score.', train_y);
% 
% train_y_pred = net(score.');
% train_acc = mean(train_y_pred == train_y);
% 
% demean_test = test_x.'-mu;
% score_test = demean_test*coeff;
% test_y_pred = net(score_test.');
% test_acc = mean(test_y_pred == test_y);
% 
% fprintf("Q3c. Rosenblatt's Perceptron with PCA\ntraining accuracy: %f, validation accuracy: %f\n", ...
%      train_acc, test_acc);

%%%%%%%%%%%%%%%%%%%   3d   %%%%%%%%%%%%%%%%%%%
fprintf("Q3d.\n");
train_acc_list = zeros(13, 1);
test_acc_list = zeros(13, 1);
hidden_neurons = [1:10, 20, 50, 100];
for i = hidden_neurons
    net = patternnet(i);
    net = train(net, train_x, train_y);

    train_y_pred = net(train_x);
    train_y_pred = train_y_pred > 0.5;
    train_acc = mean(train_y_pred == train_y);
    train_acc_list(i,:) = train_acc;
    
    test_y_pred = net(test_x);
    test_y_pred = test_y_pred > 0.5;
    test_acc = mean(test_y_pred == test_y);
    test_acc_list(i,:) = test_acc;
    fprintf("MLP batch mode (%d hidden neurons)\ntraining accuracy: %f, validation accuracy: %f\n", ...
        i, train_acc, test_acc);
end

hold on
scatter(hidden_neurons, train_acc_list, 'b');
scatter(hidden_neurons, test_acc_list, 'r');
%%%%%%%%%%%%%%%%%%%   3e   %%%%%%%%%%%%%%%%%%%

