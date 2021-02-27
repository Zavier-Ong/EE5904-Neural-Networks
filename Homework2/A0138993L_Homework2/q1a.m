clc;
clear;
close all;

x = 0;
y = 1;

cost = (1-x)^2 + 100*(y-x^2)^2;
max_epoch = 100000;
all_cost = [cost];
all_xy = [x,y;];
iteration = 0;
eta = 0.001; %learing_rate
threshold = 1e-5;

for epoch = 1:max_epoch
    iteration = iteration+1;
    %gradient descent
    old_x = x;
    old_y = y;
    x = old_x - eta*(2*(old_x-1) + 400*old_x*(old_x^2 -old_y));
    y = old_y - eta*(200*(old_y-old_x^2));
    cost = (1-x)^2 + 100*(y-x^2)^2;
    
    all_cost = [all_cost; cost];
    all_xy = [all_xy; x,y;];
    if (cost < threshold)
        break;
    end
end

num_iter = [1:iteration+1];
fprintf("Q1a. Number of iterations: %d\n", length(num_iter));
% plot
fig = figure();
subplot(3, 1, 1);
hold on;
yyaxis left;
plot(num_iter, all_xy(:,1),'--o');
ylabel('X');
yyaxis right;
plot(num_iter, all_xy(:,2),'*-');
ylabel('Y');
xlabel('Iterations');

subplot(3, 1, 2)
plot(num_iter,all_cost,'--o')
ylabel('Function Value')
xlabel('Iterations')

subplot(3, 1, 3)
plot(all_xy(:,1), all_xy(:, 2));
ylabel('Y');
xlabel('X');
%saveas(fig,sprintf('q1_images/steepest_descent.png'));
