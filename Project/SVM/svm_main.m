clc
clear

load('train.mat')
load('test.mat')

%preprocessing
%find mean of each row
mew = mean(train_data, 2);
sd = std(train_data, 0, 2);
%standardization method
stand_train = (train_data-mew)./sd;
stand_test = (test_data-mew)./sd;

%checking got kernel suitability
gram_matrix = stand_train'*stand_train;
eig_val = eig(gram_matrix)
if min(eig_val) < 0
    fprint('Since one of the eigenvalues < 0, this kernel candidate is not admissible')
    return
end




        