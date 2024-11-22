function Yn = GNTD( Y,opts )
%
%   Usage: Yn = GNTD( Y,opts );
%   Output: Yn is a ttensor
%    opts.
%         W: affinity matrix;   
%         num_of_comp: a vector specifying the dimension of Yn.core;
%         max_iter: [100] max number of iterations
%         max_in_iter: [20] max number of iterations for internal loop (for each
%               sub-nls problem)
%         tol: 1e-6 the algorithm terminates if ||A(1)-A(1)_old||<tol
%         lambda_g: hyper-parameter for graph regularization penalty

%   This code depends on the TensorToolbox which is available at:
%           http://www.sandia.gov/~tgkolda/TensorToolbox/index-2.5.html


%  If you think this algorithm is useful, please cite
%   [1] Qiu Y, Zhou G, Zhang Y, et al. Graph Regularized Nonnegative Tucker Decomposition for Tensor Data Representation[C] ...
%       ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019: 8613-8617.
%   [2] Qiu, Y., Zhou, G., Wang, Y., Zhang, Y., & Xie, S. (2020). A Generalized Graph Regularized Non-Negative Tucker Decomposition 
%       Framework for Tensor Data Representation. IEEE Transactions on Cybernetics.

%  If you have any question about the code, please contact me without any hesitate:
%  email: yuning.qiu.gd@gmail.com
%  last update: 2019/06/11

defopts=struct('W', [],'num_of_comp',[],'max_iter',500,'max_in_iter',20,'Tol',1e-2, 'lambda_g', 0);
if ~exist('opts','var')
    opts=struct();
end
[W, R, maxiter,maxiniter,tol,lambda_g]=scanparam(defopts,opts);

I=size(Y);
N=numel(I);

if isempty(R)
    error('The parameter ''NumOfComp'' must be specified.');
end

activeModes=~isinf(R);
if numel(find(activeModes==1))==1
    error('Please use NMF algorithms directly.');
end

R=round(R);
R=min(R,I);

if any(R<eps)
    error('''NumOfComp'' must be nonzero integers.');
end

% Initialization
A=cell(1,N);
AtA=cell(1,N);
AtyA=cell(1,N);

% core=tensor(zeros(R));
core=tensor(rand(R));
Ytemp=tucker_als(tensor(Y), R, 'printitn',0);

for n=1:N
    if activeModes(n)
        A{n}=Ytemp.U{n};
        % random initialisation here!
        % A{n}=rand(I(n),R(n));
        A{n}=max(A{n}, eps);
        AtA{n}=(A{n})'*A{n};
        % A{n}=rand(I(n),R(n));
    else
        A{n}=speye(I(n));
        AtA{n}=A{n};
    end
end


if lambda_g > 0
    % W = lambda_g * W;
    DCol = full(sum(W, 2));
    D = spdiags(DCol, 0, I(end), I(end));
    L = D - W;
    norm_L = lambda_g * norm(full(L));
else
    error('Lambda shoud be geater than zero');
end

Xtilde=[];

n_pos=find(activeModes,1,'last');
activeModesNames=find(activeModes==1);
for it=1:maxiter
    AN0=A{N};
    for n=1:N
        if ~activeModes(n)
            continue;
        end
        
        % update A{n}
        nindices=activeModesNames(activeModesNames~=n);
        X=ttm(core,AtA(nindices),nindices);
        core_n_T=double(tenmat(core,n,'t'));
        Xtilde=ttm(tensor(Y),A(nindices),nindices,'t');
        
        BtB=double(tenmat(X,n));
        BtB=BtB*core_n_T; % for computation convenient.
        
        YB=double(tenmat(Xtilde,n));
        YB=YB*core_n_T;
        
        if n == N 
            rho_g = 1/(norm(BtB) + norm_L);
        else
            rho = 1/norm(BtB);
        end
        alpha1=1;
        Lk=A{n};
        for init=1:maxiniter
            An0=A{n};
            if n == N
                grad = -YB + Lk*BtB + lambda_g * L* Lk;
                A{n} = max(Lk - rho_g * grad, eps);
                %A{n}=max(Lk-rho*grad-L1_Fac(n),eps);
            else
                grad = -YB + Lk*BtB;
                A{n} = max(Lk - rho * grad, eps);
            end
            Andiff = A{n}-An0;
            alpha0 = alpha1;
            alpha1 = (1+sqrt(4*alpha0^2+1))/2;
            Lk = A{n}+((alpha0-1)/alpha1)*Andiff;
        end
        
        if n == N
            nrm = max(abs(A{N}));
            nrm = max(nrm, 1e-10);
            A{N} = bsxfun(@rdivide,A{N}, nrm);
            core = ttm(core, diag(nrm), N);
        end
    
        AtA{n}=A{n}'*A{n};
    end
    
    % update core tensor
    n=n_pos;
    rho=1;
    for k=1:N
        if activeModes(k)==1
            rho=rho*norm(AtA{k});
        else 
            rho = rho * sqrt((I(k)));
        end
    end

    rho=1/rho;
    alpha1=1;
    Lk=core;
    gradY=-ttm(Xtilde,A{n},n,'t');
    for init=1:maxiniter
        core0=core;
        grad=ttm(Lk,AtA);
        grad=double(gradY+grad);
        core=double(core)-rho*grad;
        core=max(core,eps);
        corediff=core-core0;
        alpha0=alpha1;
        alpha1=(1+sqrt(4*alpha0^2+1))/2;
        Lk=tensor(core+((alpha0-1)/alpha1)*corediff);
    end
    core=tensor(core);
   
    % stoping criterion
    if it>20
        if max(abs(A{N}-AN0))<tol
            break;
        end
    end
end

Yn=ttensor(tensor(core),A);
end