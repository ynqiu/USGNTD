clear;
addpath(genpath(pwd));

% load data
load('ORL_32x27.mat')
rand('twister', 5489);

% parameter setting
repeatTimes=10;
r1 = 22;
r2 = 21;
k = 10;

whichGraph = 1;
kNeighbors = 5;
lambda_g=1e-1;
R = [r1, r2, k];

acc_array = zeros(1,repeatTimes);
nmi_array = zeros(1,repeatTimes);

for i=1:repeatTimes
    % R = [r1, r2, inf, k]; if Y is a color image data set.
    W = similarityW(fea', whichGraph, kNeighbors);
    fea = NormalizeFea(double(fea));
    Y =reshape(fea', [32, 27,400]);
    
    N = numel(size(Y));

    opts = struct('W', W,'num_of_comp', R,'max_iter', 500,'max_in_iter', 20, 'Tol',1e-2, 'lambda_g', lambda_g);

    % Please repeat 20 times to obtain the average accuracy and standard
    % derivation.
    [Ydec] = GNTD(double(Y), opts); 

    clusterResults = evalResults(Ydec.U{N}', gnd);
    % clusterResults = [accuracy, nmi, purity];
    % fprintf('\nAC=%.2f, NMI=%.2f, Purity=%.2f',clusterResults(1), clusterResults(2), clusterResults(3));
    acc_array(i) = clusterResults(1);
    nmi_array(i) = clusterResults(2);
end
av_acc = mean(acc_array);
av_nmi =mean(nmi_array);

fprintf('\naverage AC=%.2f, average NMI=%.2f\n', av_acc*100, av_nmi*100);

% av_acc = mean(acc_array);
% av_nmi =mean(nmi_array);
% 
% fprintf('\naverage AC=%.2f, average NMI=%.2f\n', av_acc*100, av_nmi*100);

function W =  similarityW(Y, whichGraph, kNeighbors)
    options = [];
    switch whichGraph
    case 1
        options.k = kNeighbors;
        options.WeightMode = 'HeatKernel';
        W = constructW(Y', options);
    case 2
        options.k = kNeighbors;
        options.WeightMode = 'Binary';
        W = constructW(Y', options);
    end
end

