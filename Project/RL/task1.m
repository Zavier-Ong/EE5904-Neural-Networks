clc
clear
close all

load('task1.mat');
threshold = 0.005;
discount_factor = [0.5, 0.9];
exploration_mode = [1, 2, 3, 4];
max_trials = 3000;
num_runs = 10;
[states, actions] = size(reward);

for i=1:size(discount_factor, 2)
    for j = 1:size(exploration_mode, 2)
        mode = exploration_mode(j);
        for run = 1: num_runs
            fprintf('Run %d started.', run);
            %start timer
            tic;
            %initialize parameters
            trial = 1;
            start_pos = 1;
            end_pos = 100;
            Q = zeros(states, actions);
            converged = 0;
            while trial <= max_trials && converged ~= 1
                discount = discount_factor(i);
                k = 1;
                sk = start_pos;
                old_Q = Q;
                while sk ~= end_pos
                    explore_rate = select_explore_mode(mode, k);
                    ak = apply_action
                end
                
            end
                
        end
    end
end


%% Functions
function explore_rate = select_explore_mode(mode, k)
    switch mode
        case 1
            explore_rate = 1/k
        case 2
            explore_rate = 100 / (100 + k)
        case 3
            explore_rate = (1 + log(k)) / k
        case 4
            explore_rate = (1 + 5*log(k)) / k
    end
end
