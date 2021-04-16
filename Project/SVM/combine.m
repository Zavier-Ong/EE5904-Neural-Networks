clear all
close all
clc
%% loading data
% Create example data
z = load('train.mat');
y = load('test.mat');
eval_data = [y.test_data, z.train_data];
eval_label = [y.test_label; z.train_label];

idx = randperm(3536);

x.eval_data = eval_data(:,idx(1:600));
x.eval_label = eval_label(idx(1:600), 1);
% Save result in a new file
save('eval.mat','-struct','x')