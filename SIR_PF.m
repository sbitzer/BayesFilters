% function [Pstore,Neff] = SIR_PF(Y,T,P0,ffun,gfun,Q,R,options)
%
% implementation of sequential importance resampling (the standard variant
% of a particle filter) based on
%       Doucet, A. & Johansen, A. M. 
%       A Tutorial on Particle Filtering and Smoothing: Fifteen years later
%       Oxford Handbook of Nonlinear Filtering
%       Oxford University Press, 2011
% and wikipedia (the bit with resampling only when the number of effective
% particles is too low)
%
% the model is
%       dx/dt = f(x) + w
%       y = g(x) + v
% where w, v are Gaussian random variables
%
% continuous dynamics is numerically integrated using Euler-Maruyama
% 
% this implementation does batch filtering (all data points given to the
% function, but no smoothing is done)
%
% in:
%       Y       -   the data points to be filtered
%                   [ny,nt] = size
%       T       -   time points at which data were sampled
%                   [1,nt] = size
%       P0      -   upper and lower bounds for initial particle
%                   distribution, i.e., particles are initialised by
%                   drawing from a uniform distribution within the bounds
%                   given by P0
%                   [nx,2] = size, P0(:,1) lower bound, P0(:,2) upper bound
%       ffun    -   handle to dynamics function dx/dt = f(x,fopt)
%       gfun    -   handle to observation function y = g(x,gopt)
%       Q       -   prior dynamics covariance (covariance of w)
%                   [nx,nx] = size
%       R       -   prior observation covariance (covariance of v)
%                   [ny,ny] = size
%       options -   structure with fields:
%             .np - number of particles to use
%           .fopt - options/parameters passed to ffun
%                   [default: empty struct]
%           .gopt - options/parameters passed to gfun
%                   [default: empty struct]
%             .dt - step size for Euler-Maruyama integration of dynamics
%                   [default: .1]
%           .neff - threshold for effective number of samples, (resample, 
%                   if effective number of samples drops below neff)
%                   [default: 2/3 * np]
%
% Copyright (C) 2015 Sebastian Bitzer
function [Pstore,Neff] = SIR_PF(Y,T,P0,ffun,gfun,Q,R,options)

nt = length(T);
nx = size(P0,1);

np = options.np;

Qchol = chol(Q, 'lower');
Rinv = inv(R);
Rdet = det(R);

if isnonemptyfield(options,'fopt')
    fopt = options.fopt;
else
    fopt = struct([]);
end
if isnonemptyfield(options,'gopt')
    gopt = options.gopt;
else
    gopt = struct([]);
end
if isnonemptyfield(options,'dt')
    dt = options.dt;
else
    dt = .1;
end
if isnonemptyfield(options,'neff')
    neff = options.neff;
else
    neff = 2/3 * np;
end

% sample particles from uniform prior distribution of hidden variables
P = bsxfun(@plus, bsxfun(@times, rand(nx,np), diff(P0,1,2)), P0(:,1));

% initial importance weights
W = ones(np,1) / np;

% for printing progress
fprintf('filtering completed:   0%%')
deletePast = sprintf('\b')*ones(1,4);

Dt = diff([0,T]);
Pstore = nan(nx,np,nt);
Neff = nan(1,nt);
for t = 1:nt
    % determine step size for integration until current measurement
    nsteps = ceil(Dt(t) / dt);
    dtt = Dt(t) / nsteps;
    
    % drawing samples from the proposal distribution
    % (simple Euler-Maruyama integration)
    dW = sqrt(dtt) * randn(nx,np,nsteps);
    for it = 1:nsteps
        dP = ffun(P, fopt);
        
%         % no noise
%         P = P + dtt * dP;
        
        % noise with prior covariance
        P = P + dtt * dP + Qchol * dW(:,:,it);
    end
    
    % update importance weights using the transition prior as importance
    % (function, note that P contains the different means, but as it occurs
    % inside a square term, the order of P and Y doesn't matter)
    pY = mvnormpdf(gfun(P,gopt)', Y(:,t)', [], Rinv, Rdet);
    W = W .* pY;
    
    % normalise the weights
    W = W / sum(W);
    
    Neff(t) = 1/sum(W.^2);
    if Neff(t) < neff
        % resampling using Kitagawa's systematic (deterministic) resampling
        % based on Arnaud Doucet's and Nando de Freitas' deterministicR.m
        % is it faster using Matlab's built-ins? - they potentially implement
        % many more loops through 1:np, though, so I kept this formulation
        U = (rand(1) + (0:np-1)') / np;
        Wcum = cumsum(W);

        N = zeros(np,1);
        j = 1;
        for p = 1:np
            while U(p) > Wcum(j)
                j = j + 1;
            end
            N(j) = N(j) + 1;
        end

        ind = nan(np,1);
        n0 = 0;
        for p = 1:np
            ind(n0 + (1:N(p))) = p;
            n0 = n0 + N(p);
        end

        W = ones(np,1) / np;
        P = P(:,ind);
    end
    
    Pstore(:,:,t) = P;
    
    %% printing progress
    % only print if full per cents changed
    if diff(floor([t-1,t]/(nt/100)))>=1
        fprintf('%s%3d%%',deletePast,round(t/nt*100));
    end
end

fprintf('\n');  % end of printing progress
