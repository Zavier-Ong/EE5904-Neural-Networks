clc
clear
close all

x_train = -2:0.05:2;
x_test = -3:0.01:3;

y_train = mlp_approx_fn(x_train);
y_test = mlp_approx_fn(x_test);

hidden_neurons = [1:10, 20, 50, 100];
%hidden_neurons = [10];
epochs = 10;
for i = hidden_neurons
    % training
    net = train_batch(i,x_train,y_train);
    % test
    y_out = net(x_test);
    % plot
    fig = figure();
    hold on
    plot(x_test,y_test,'-','LineWidth',2)
    plot(x_test,y_out,'-','LineWidth',2)
    ylim([-2.5 2.5])
    legend('Desired output', 'MLP output')
    title(sprintf('MLP: 1-%d-1 (Batch Mode (trainlm))', i))
    ylabel('Y')
    xlabel('X')
    hold off
    saveas(fig,sprintf('q2_images/lm_%d.png',i));
    
    %outputs of MLP x=-3 and x-3
    if (i==10)
        %x=-3
        y_desired = mlp_approx_fn(-3);
        y_out = net(-3);
        fprintf('(x = -3) desired: %d output: %d\n', y_desired, y_out);
        %x=3
        y_desired = mlp_approx_fn(3);
        y_out = net(3);
        fprintf('(x = 3)  desired: %d output: %d\n', y_desired, y_out);
    end
end


function y = mlp_approx_fn(x)
    y = 1.2*sin(pi*x) - cos(2.4*pi*x);
end

function net = train_batch( n, images, labels)
    % 1. Change the input to cell array form for sequential training
    images_c = num2cell(images, 1);
    labels_c = num2cell(labels, 1);
    
    % 2. Construct and configure the MLP
    net = feedforwardnet(n, 'trainlm');
    net = train(net, images_c, labels_c);
end