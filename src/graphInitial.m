function W = graphInitial(opts)
% 
% Build semi-supervised affinity matrix via constrained propogation
%
defopts = struct( 'gnd', [], 'W', [], 'maxIter', 200, ...
    'solver', 'closedform', 'alp', 0.5, 'cannotKnownPercentage',0.1, 'mustKnownPercentage', 0.1);
if ~exist('opts', 'var')
    opts = struct();
end
[gnd, W, maxIter, solver, alp, cannotKnownPercentage, mustKnownPercentage] = scanparam(defopts, opts);

nSmp = numel(gnd);
% initialize the graph without label
% options = struct('WeightMode', 'Heatkernel', 'k', 5);
% W = constructW(fea, options);

% construct Laplacian matrix L
DCol = full(sum(W, 2));
DCol = 1 ./ (sqrt(DCol));
D = spdiags(DCol, 0, nSmp, nSmp);
% L = eye(nSmp, nSmp) - D * W * D;
L = D * W * D;
% construct the matrix Z with label inforation

[~, ~, fullIndex] = unique(gnd);
% numOfCate = numel(categories);
% if gnd is not full konwn
% fullIndex(fullIndex == 1) = 0;
[indexM, indexN] = meshgrid(fullIndex, fullIndex');
[indexM, indexN] = deal(reshape(indexM, [], 1), reshape(indexN, [], 1));

% cannot link mapping
cannotLink = indexM ~= indexN;
% only a little percentage link are preserved

% must link mapping
allSameLink = indexM == indexN;
zeroLink = indexM == 0;
mustLink = allSameLink - zeroLink;
mustLink = mustLink == 1;

%  matrix Z construction
Z = zeros(nSmp, nSmp);
Z(mustLink) = 1;
Z(cannotLink) = -1;
% normalized: diag(Z) = 0;
Z = Z + diag(-diag(Z));

Ztril = tril(Z);
cannotLinkPartIndex = find(Ztril == -1);
mustLinkPartIndex = find(Ztril == 1);

nCannotLink = numel(cannotLinkPartIndex);
nMustLink = numel(mustLinkPartIndex);

unknownSamples = randperm(nCannotLink, round(cannotKnownPercentage*nCannotLink));
unknownSamples = sort(unknownSamples);
cannotLinkPartIndex(unknownSamples) = [];
Ztril(cannotLinkPartIndex) = 0;

% must link reduction
unknownSamples = randperm(nMustLink, round(mustKnownPercentage*nMustLink));
unknownSamples = sort(unknownSamples);

% mustLinkPartIndex(unknownSamples) = 0;
mustLinkPartIndex(unknownSamples) = [];
Ztril(mustLinkPartIndex) = 0;

Z = Ztril + Ztril';

% update the matrix F
switch upper(solver)
case upper('iterative')
    Fv = eye(nSmp, nSmp);
    for iter = 1 : maxIter
        Fv0 = Z;
        Fv = alp * L * Fv + (1-alp) * Z;

        if iter > 50
            if norm(Fv0 - Fv) < eps
                break;
            end
        end
    end

    Fh = Fv;
    for iter = 1 : maxIter
        Fh0 = Fh;
        Fh = alp * Fh * L + (1 - alp) * Fv;

        if iter > 50
            if norm(Fh0 - Fh) < eps
                break;
            end
        end
    end
    Fopt = Fh;

case upper('closedform')
    Fopt = (1 - alp)^2 * ( (( eye(nSmp, nSmp) - alp * L) \ Z)  / (eye(nSmp, nSmp) - alp * L));
end

negZIndex = Z==-1;
posZIndex = Z==1;
Fopt(negZIndex) = -1;
Fopt(posZIndex) = 1;

leqZeroIndex = Fopt<0;
Fopt(leqZeroIndex) = 0;

W = Fopt - diag(diag(Fopt));
end

