clear;
clc;

x = [1 0.5; 1 1.5; 1 3; 1 4.0; 1 5.0];
d = [8.0, 6.0, 5, 2, 0.5];
rng('default');
s = rng;

weights = randn(1, 2);
trajectory = weights;
l_rate = 0.002;
total_epochs = 2000;

for epoch = 1:total_epochs
    for i = 1:length(x(:,1))
        if (x(i, 1:2)*weights') == d(i)
            continue
        else
            err = d(i) - x(i, 1:2)*weights';
            weights = weights + l_rate*err*x(i, 1:2);
            trajectory = [trajectory;weights];
        end
    end
end

y_lms = x*weights';
w = (x'*x) \ x' *d';
y_lls = x*w;
disp(weights);

fig1 = figure;
title('Fitting result (LMS vs \color{red}LLS\color{black})');
%title('Fitting result of LMS');
xlabel('x');
ylabel('y');
hold on;
scatter(x(:, 2), d, 'filled');
plot(x(:, 2), y_lms, 'k');
plot(x(:,2), y_lls, 'r');
hold off;
saveas(fig1, sprintf('4d_fit_%.3f_%d.png', l_rate, total_epochs));

fig2 = figure;
hold on;
title('Trajectory of weights');
xlabel("Learning Steps");
ylabel("Weights");
plot(0:length(trajectory)-1,trajectory(:,1))
plot(0:length(trajectory)-1,trajectory(:,2))
legend({'w0','w1'})
hold off;
saveas(fig2, sprintf('4d_traj_%.3f_%d.png', l_rate, total_epochs));