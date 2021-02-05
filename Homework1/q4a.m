clc;
clear;

x = [1 0.5; 1 1.5; 1 3; 1 4.0; 1 5.0];
d = [8.0; 6.0; 5; 2; 0.5];

%inv(x'*x) * x' == (x'*x) \ x'
w = (x'*x) \ x' *d;
y = x*w;

fig1 = figure;
xlabel('x');
ylabel('y');
title("Fitting result of LLS");
hold on;
scatter(x(:,2), d, 'filled')
p = plot(x(:,2), y);
disp(w);
%saveas(fig1, 'q4a.png');
