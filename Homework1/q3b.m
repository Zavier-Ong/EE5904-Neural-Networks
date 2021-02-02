clc
clear

and = [1 1 1 1; 0 0 1 1; 0 1 0 1; 0 0 0 1];
or = [1 1 1 1; 0 0 1 1; 0 1 0 1; 0 1 1 1];
nand = [1 1 1 1; 0 0 1 1; 0 1 0 1; 1 1 1 0];
complement = [1 1; 0 1; 1 0];


func = complement;
l_rate = 0.1;
[dim, num_inputs] = size(func);
%setting seed to 0 to ensure reproducibility
rng('default')
s = rng

weights = rand(1, dim-1);
count = 1

classified = false;

while ~(classified)
    classified = true;
    for i = 1: num_inputs
        y = (dot(weights(count, 1:dim-1), func(1:dim-1, i)) >= 0);
        err = func(dim, i) - y;
        if (~(err == 0))
            classified = false;
            weights(count+1, 1:dim-1) = weights(count, 1:dim-1) + (l_rate * err * func(1:dim-1, i))';
            count = count + 1;
        end
    end
end

if dim-1 == 2 %handling complement case
    figure;
    title('Trajectory of weights');
    hold on;
    xlabel("Iteration count");
    ylabel("Weight");
    plot(weights(1:size(weights, 1), 1));
    plot(weights(1:size(weights, 1), 2));
    legend({'w0', 'w1'});
    grid on;
    hold off;
    
    figure;
    title('Decision Boundary (Learning procedure vs off-line calculations)')
    hold on;
    xlim([0,2])
    ylim([0,2])
    %plot truth table
    for i=1:num_inputs
        if func(end, i) == 1
            plot(func(end, i),0, 'b^');
        else
            plot(func(end, i),0, 'ro');
        end
    end
    %plot learning procedure
    x = -weights(end,1)/weights(end,2);
    xline(x,'k');
    %plot COMPLEMENT off line calculation
    xline(0.5, 'r');
    xlabel("x1");
    ylabel("x2");
    grid on
    hold off
end        

if dim-1 == 3
    figure;
    title('Trajectory of weights');
    hold on;
    xlabel("Iteration count");
    ylabel("Weight");
    plot(weights(1:size(weights,1),1));
    plot(weights(1:size(weights,1),2));
    plot(weights(1:size(weights,1),3));
    legend({'w0','w1','w2'});
    grid on
    hold off
    
    figure;
    title('Decision Boundary (Learning procedure vs off-line calculations)')
    hold on;
    x = -10:100;
    m = -weights(end,2)/weights(end,3);
    c = -weights(end,1)/weights(end,3);
    xlim([0,2])
    ylim([0,2])
    %plot truth table
    for i = 1:num_inputs
        if func(end,i) == 1
            plot(func(2,i),func(3,i),'b^');
        else
            plot(func(2,i),func(3,i),'ro');
        end
    end
    %plot learning procedure
    y = m * x + c;
    plot(x, y,'k')
    %plot off line calculation
    y_off_and = -x + 1.5;
    y_off_or = -x + 0.5;
    y_off_nand = -x + 1.5;
    plot(x, y_off_nand, 'r');
    xlabel("x1");
    ylabel("x2");
    grid on
    hold off
end