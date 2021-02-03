clear;
clc;

x = [1 0.5; 1 1.5; 1 3; 1 4.0; 1 5.0];
d = [8.0, 6.0, 5, 2, 0.5];
rng('default');
s = rng;

weights = randn(1, 2);
trajectory = weights;
l_rate = 0.001;

for epoch = 1:100
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

y = x*weights';

figure
hold on 
scatter(x(:, 2), d, 'filled')
plot(x(:, 2), y);
hold off

figure
hold on 
xlabel("Learning Steps");
ylabel("Weights");
plot(0:length(trajectory)-1,trajectory(:,1))
plot(0:length(trajectory)-1,trajectory(:,2))
legend({'w0','w1'})
hold off