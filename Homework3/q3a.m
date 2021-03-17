clc
clear
close all

%training points sampled from sine curve
x = linspace(-pi, pi, 400);
train_x = [x; 2*sin(x)]; %2x400 matrix

%SOM
T = 600;
N = 1;
M = 36;
lr0 = 0.1;
sigma0 = sqrt(M^2*N^2) / 2;
tau = T / log(sigma0);
weights = rand(2, 36);
for n = 1:T
    lr = lr0*exp(-n/T);
    sigma = sigma0*exp(-n/tau);
    %sample input vector
    i = randperm(400, 1);
    %determine winner
    distance = sum((train_x(:,i) - weights).^2,1);
    [~, winner] = min(distance, [], 2);
    neuron_position = (1:36);
    d = abs(neuron_position - winner);
    h = exp(-d.^2/(2*sigma^2));
    % Update
    weights = weights + lr*h.*(train_x(:,i) - weights);
end

fig = figure();
hold on
plot(train_x(1,:), train_x(2,:), '+r');
plot(weights(1,:), weights(2,:), '-ok');
hold off
saveas(fig, 'q3a.png');