clc
clear
close all

x_train = -2:0.05:2;
x_test = -3:0.01:3;

y_train = mlp_approx_fn(x_train);
y_test = mlp_approx_fn(x_test);

hidden_neurons = [1:10, 20, 50, 100];
%hidden_neurons = [10];
epochs = 200;
for i = hidden_neurons
    % training
    net = train_seq(i,x_train,y_train,81,epochs);
    % test
    y_out = net(x_test);
    % plot
    fig = figure();
    hold on
    plot(x_test,y_test,'-','LineWidth',2)
    plot(x_test,y_out,'-','LineWidth',2)
    ylim([-2.5 2.5])
    legend('Desired output', 'MLP output')
    title(sprintf('MLP: 1-%d-1 (Sequential Mode)', i))
    ylabel('Y')
    xlabel('X')
    hold off
    saveas(fig,sprintf('q2_images/sequential_%d.png',i));
    
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

function net = train_seq( n, images, labels, train_num, epochs )
    % 1. Change the input to cell array form for sequential training
    images_c = num2cell(images, 1);
    labels_c = num2cell(labels, 1);
    
    % 2. Construct and configure the MLP
    net = fitnet(n);
    net.divideFcn = 'dividetrain'; % input for training only
    net.performParam.regularization = 0.25; % regularization strength
    net.trainFcn = 'traingdx'; % 'trainrp' 'traingdx'
    net.trainParam.epochs = epochs;
    
    % 3. Train the network in sequential mode
    for i = 1 : epochs
        display(['Epoch: ', num2str(i)])
        idx = randperm(train_num); % shuffle the input
        net = adapt(net, images_c(:,idx), labels_c(:,idx));
    end
end