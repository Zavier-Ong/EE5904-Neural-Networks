clc
clear
close all

x = 0;
y = 1;

cost = (1-x)^2 + 100*(y-x^2)^2;
max_epoch = 100000;
all_cost = [cost];
all_xy = [x,y;];
iteration = 0;
threshold = 1e-5;

for epoch = 1:max_epoch
    iteration = iteration+1;
    g = [(2*(x-1)+400*x*(x^2 -y)); (200*(y-x^2))];
    H = [(1200*x^2-400*y+2), (-400*x); (-400*x), 200];
    w = [x;y]- H\g; %inv(H)*g = H\g
    
    x = w(1);
    y = w(2);
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
saveas(fig,sprintf('q1_images/newton.png'));
    