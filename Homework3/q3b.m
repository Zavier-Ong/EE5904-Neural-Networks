clc
clear
close all

rng(3);
X = randn(800,2);
s2 = sum(X.^2,2);
train_x = (X.*repmat(1*(gammainc(s2/2,1).^(1/2))./sqrt(s2),1,2))';

%SOM
T = 36*600;
lr0 = 0.1;
sigma0 = sqrt(6^2*6^2) / 2;
tau = T / log(sigma0);
weights = rand(2, 6, 6);

for n = 1:T
    lr = lr0*exp(-n/T);
    sigma = sigma0*exp(-n/tau);
    %sample input vector
    i = randperm(800, 1);
    %determine winner
    distance = squeeze(sum((train_x(:,i) - weights).^2,1))';
    [~,winner] = min(distance,[],'all','linear');
    row = ceil(winner/6);
    col = mod(winner,6);
    if col == 0
        col = 6;
    end
    %get time-carying neighborhood function
    neuron_position = (1:6);
    d_j = (neuron_position - col).^2;
    d_i = (neuron_position - row).^2;
    dji = d_j' + d_i;
    h = exp(-dji./(2*sigma^2)); 
    % Update
    h = permute(repmat(h,[1,1,2]),[3 2 1]);
    weights = weights + lr*h.*(train_x(:,i) - weights);
end

fig = figure();
hold on
plot(train_x(1,:), train_x(2,:), '+r');
weights_1 = squeeze(weights(1, :, :));
weights_2 = squeeze(weights(2, :, :));
for i = 1:6
    plot(weights_1(i,:), weights_2(i,:), 'bo-');
    plot(weights_1(:,i), weights_2(:,i), 'bo-');
end
hold off  
saveas(fig, 'q3b.png');