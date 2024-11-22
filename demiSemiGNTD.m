clear;
rand('twister', 5489);
addpath(genpath(pwd));

% load data
load('USPS50.mat')

% parameter setting
repeatTimes=10;
r1 = 8;
r2 = 8;
k = 10;

lambda_g=1e-1;
R = [r1, r2, k];
% R = [r1, r2, inf, k]; if Y is a color image data set.
fea = fea+1;
fea = NormalizeFea(double(fea));

acc_array = zeros(1,repeatTimes);
nmi_array = zeros(1,repeatTimes);

for i=1:repeatTimes
    % construct the graph
    options.k=5;
    options.WeightMode='HeatKernel';
    Wun = constructW(fea, options);
    % propagate the graph matrix
    optsW = struct( 'gnd', gnd, 'W', Wun, 'maxIter', 200, ...
            'solver', 'closedform', 'alp', 0.5, 'cannotKnownPercentage',0.05, 'mustKnownPercentage', 0.05);
    W = graphInitial(optsW);
    % tensorizing the data set
    Y =reshape(fea', [16, 16, 500]);
    N = numel(size(Y));
    % run SGNTD
    opts = struct('W', W,'num_of_comp', R,'max_iter', 500,'max_in_iter', 20, 'Tol',1e-2, 'lambda_g', lambda_g);
    [Ydec] = GNTD(double(Y), opts); 
    clusterResults = evalResults(Ydec.U{N}', gnd);
    % fprintf('\nAC=%.2f, NMI=%.2f, Purity=%.2f',clusterResults(1), clusterResults(2), clusterResults(3));
    acc_array(i) = clusterResults(1);
    nmi_array(i) = clusterResults(2);
end

av_acc = mean(acc_array);
av_nmi =mean(nmi_array);

fprintf('\naverage AC=%.2f, average NMI=%.2f\n', av_acc*100, av_nmi*100);