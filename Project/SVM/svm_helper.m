classdef svm_helper
    methods(Static)
        function isSuitable = check_mercer(matrix)
            e = eig(matrix);
            if min(e) < -1e-4
                isSuitable = 0;
            else
                isSuitable = 1;
            end

        end

        function acc = get_linear_kernel_acc(wo, bo, data, label)
            g = wo'*data+bo;
            pred_label = sign(g)';
            acc = mean(pred_label==label)*100;
        end
        
        function acc = get_poly_kernel_acc(alpha, bo, p, train_data, train_label, data, label)
            K = (train_data'*data+1).^p;
            adk_term = alpha.*train_label.*K;
            adK = sum(adk_term, 1)';
            g = sign(adK +bo);
            pred_label = sign(g);
            acc = mean(pred_label == label)*100;
        end
    end
end


        