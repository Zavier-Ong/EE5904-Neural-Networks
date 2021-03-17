clc
clear
close all
% Matric A0138993L
% Classes chosen: 9 and 3
load('characters10.mat');
% set seed for reproducibility
rng(234);

train_idx = find(train_label ~= 3 & train_label ~= 9);
TrLabel = train_label(train_idx);
train_x = train_data(train_idx, :);
% normalizing train data
train_x = mat2gray(train_x(:,:))';

T = 1000;
lr0 = 0.1;
sigma0 = sqrt(10^2*10^2) / 2;
tau = T / log(sigma0);
weights = rand(784, 10, 10);

for n = 1:T
    lr = lr0*exp(-n/T);
    sigma = sigma0*exp(-n/tau);
    %sample input vector
    i = randperm(2400, 1);
    %determine winner
    distance = squeeze(sum((train_x(:,i) - weights).^2,1))';
    [~,winner] = min(distance,[],'all','linear');
    [col, row] = ind2sub(size(distance), winner);
    
    %get time-carying neighborhood function
    neuron_position = (1:10);
    d_j = (neuron_position - col).^2;
    d_i = (neuron_position - row).^2;
    dji = d_j' + d_i;
    h = exp(-dji./(2*sigma^2)); 
    % Update
    h = permute(repmat(h,[1,1,784]),[3 2 1]);
    weights = weights + lr*h.*(train_x(:,i) - weights);
end

fig = figure();
fig.Position = [100 100 1200 800];
sgtitle('Trained weights of output neurons');
marked_neuron = zeros(10);

for i = 1:size(weights, 2)
    for j = 1:size(weights, 3)
        distance = squeeze(sum((train_x(:,:)-weights(:,i,j)).^2, 1))';
        [~, win_idx] = min(distance, [], 'all', 'linear');
        winner_label = TrLabel(win_idx);
        marked_neuron(i, j) = winner_label;
        subplot(10,10, ((i-1)*10+j));
        imshow(reshape(weights(:,i,j), 28, 28));
    end
end
saveas(fig, 'q3c1.png');