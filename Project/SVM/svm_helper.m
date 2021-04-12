classdef svm_helper
    methods(Static)
        function check_mercer(matrix)
            e = eig(matrix);
            if min(e) < -1e-4
                fprintf('Kernel candidate is not admissible\n');
                return;
            else
                fprintf('Kernel candidate is admissible\n');
                return;
            end

        end

        function acc = get_linear_kernel_acc(wo, bo, data, label)
            g = wo'*data+bo;
            pred_label = sign(g)';
            acc = mean(pred_label==label);
        end
        
        function acc = get_poly_kernel_acc(alpha, bo, p, train_data, train_label, data, label)
            K = (train_data'*data+1).^p;
            adk_term = alpha.*train_label.*K;
            adK = sum(adk_term, 1)';
            g = sign(adK +bo);
            pred_label = sign(g);
            acc = mean(pred_label == label);
        end
    end
end


        