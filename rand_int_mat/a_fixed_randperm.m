clear;
repeat_times =20;
%%
% COIL20
datasets = 'ORL';
num_of_class = 40;
gap = 1;
start_k = 3;
%%
k_array = start_k: gap : num_of_class;

%%
for j = 1:1
    k = k_array(j);
    % generate the k class samples
    rand_int_mat = [];
    for i = 1: repeat_times
        rand_int = randperm(num_of_class, k);
        rand_int_mat = [rand_int_mat; rand_int];
    end
    mat_name = [datasets, '_', 'k', '_',num2str(k), '.mat'];
    save(mat_name, 'rand_int_mat');
end