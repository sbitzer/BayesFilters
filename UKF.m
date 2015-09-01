% [mX, P, peY, peX, mYpred, mXpred, Ppred, Pypred, K, nposdeferr] = ...
%            UKF(Y, T, x0, P0, ffun, gfun, Q, R, options)
%
% adaptive UKF based on allowing multiplicative noise by letting the model
% functions cope with noise and just applying the standard unscented
% transform to a state augmented with the noise variables, this is the most
% general formulation of the UKF and may therefore also be used with
% non-multiplicative noise by simply providing corresponding model
% functions.
% 
% Implements the standard unscented Kalman filter as described in:
%     Wan, E. A. & van der Merwe, R. 
%     The Unscented Kalman Filter
%     in: Haykin, S. (ed.)
%     Kalman Filtering and Neural Networks
%     John Wiley & Sons, Inc., 2001
% but adds a simple Euler integration step such that it can be used with
% continuous dynamics. In particular, it is assumed that the dynamics
% function ffun returns dx instead of directly x_t+1 (see description of
% ffun below).
%
% in:
%       Y       -   observed data points for all nt data points
%                   [ny,nt] = size
%       T       -   time points at which the data points were sampled,
%                   assuming that they are ordered and T(1) > 0 (the first
%                   data point was sampled shortly after the clock started
%                   running)
%                   [1,nt] = size
%       x0      -   mean of prior density over hidden states
%                   [nx,1] = size
%       P0      -   covariance matrix of prior density over hidden
%                   states
%                   [nx,nx] = size
%       ffun    -   dynamics function (handle)
%                       dx = ffun(x, w, fopt)
%                   where x is the current state, w is the current noise
%                   and fopt contains parameters of ffun, it is assumed
%                   that ffun defines a differential equation such that
%                   dx/dt = f(x), simple Euler integration is used to
%                   obtain x_t+1 from x_t and f
%       gfun    -   observation function (handle)
%                       y = gfun(x, v, gopt)
%       Q       -   covariance matrix of the transition density, note that
%                   this is defined as the covariance of the noise process
%                   after t=1 time units and is independent of the
%                   integration step size dt, to implement this, Q is
%                   divided by dt before it used in the discretised
%                   numerical integration within the filter prediction step
%                   [nx,nx] = size
%       R       -   covariance matrix of the observation density
%                   [ny,ny] = size
%       options -   parameter structure with fields:
%           .fopt - parameters of the dynamics function
%                   [default: []]
%           .gopt - parameters of the observation function
%                   [default: []]
%          .alpha - alpha-parameter of UKF
%                   [default: 1]
%          .kappa - kappa-parameter of UKF
%                   [default: 3 - (2*nx + ny)]
%           .beta - beta-parameter of UKF
%                   [default: 0]
%             .dt - step size for Euler integration
%                   [default: .1]
% out:
%       mX      -   means of posterior state densities
%                   [nx, nt] = size
%       P       -   covariance matrices of posterior state densities
%                   a cell containing [nx,nx]-sized matrices, depends on dt
%                   as further described below for Ppred
%                   [1, nt] = size
%       peY     -   observation prediction errors
%                   [ny, nt] = size
%       peX     -   state prediction errors (observation prediction errors
%                   transformed by Kalman gain)
%                   [nx, nt] = size
%       mYpred  -   means of the observation prediction densities
%                   [ny, nt] = size
%       mXpred  -   means of the dynamics prediction densities
%                   [nx, nt] = size
%       Ppred   -   covariances of the dynamics prediction densities
%                   a cell containing [nx,nx]-sized matrices
%                   note that Ppred depends on the size of the time steps
%                   defined in T; in particular, Ppred will contain a 
%                   contribution from Q, but this will be in the order of 
%                   N*dt*Q where N is the number of steps of size dt which
%                   need to be made when going from time T(i) to T(i+1)
%                   [1, nt] = size
%       Pypred  -   covariances of the observation prediction densities
%                   a cell containing [ny,ny]-sized matrices
%                   [1, nt] = size
%       K       -   cell array of Kalman gain matrices with dims [nx, ny]
%                   [1, nt] = size
%   nposdeferr  -   number of errors caught, because the estimated
%                   covariance matrix of the posterior state density was
%                   not positive definite, to fix the error the covariance
%                   matrix is made positive definite by setting all
%                   eigenvalues <=0 to 1e-15, should ideally be 0, any
%                   value above 0 may indicate numerical problems which may
%                   make the results unreliable
% author:
%       Copyright (C) 2015 Sebastian Bitzer
function [mX, P, peY, peX, mYpred, mXpred, Ppred, Pypred, K, nposdeferr] = ...
            UKF(Y, T, x0, P0, ffun, gfun, Q, R, options)

%% extract options and parameters
[ny,nt] = size(Y);
nx = numel(x0);
nxa = 2 * nx + ny;  %   dimension of state augmented with noise variables

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
if isnonemptyfield(options,'alpha')
    alpha = options.alpha;
else
    alpha = .01;
end
if isnonemptyfield(options,'kappa')
    kappa = options.kappa;
else
    kappa = 3 - nxa;
end
if isnonemptyfield(options,'beta')
    beta = options.beta;
else
    beta = 2;
end
if isnonemptyfield(options,'dt')
    dt = options.dt;
else
    dt = .1;
end
if isnonemptyfield(options, 'verbose')
    verbose = options.verbose;
else
    verbose = 1;
end


%% initialise
% constants
lambda = alpha^2 * (nxa + kappa) - nxa;
sqrtc = sqrt(alpha^2 * (nxa + kappa));

% unscented transform weights
wm = [lambda/(nxa + lambda); ...
      ones(2*nxa,1) / (2*nxa+2*lambda)];
wc = [lambda/(nxa + lambda) + 1 - alpha^2 + beta;...
      ones(2*nxa,1) / (2*nxa+2*lambda)];
Wc = diag(wc);

% determine sqrt of initial covariance using lower-triangular chol
A = chol( blkdiag(P0, (1/dt) * Q, R), 'lower');

% initial sigma points
XS = bsxfun(@plus, [x0; zeros(nx+ny,1)], sqrtc*[zeros(nxa,1) A -A]);


%% loop over observations
% for printing progress
if verbose
    fprintf('filtering completed:   0%%')
    deletePast = sprintf('\b')*ones(1,4);
end

Dt = diff([0,T]);
mX = nan(nx,nt);
P = cell(1,nt);
K = cell(1,nt);
mXpred = nan(nx,nt);
mYpred = nan(ny,nt);
Ppred = cell(1,nt);
Pypred = cell(1,nt);
peX = nan(nx,nt);
peY = nan(ny,nt);
nposdeferr = 0;
for t = 1:nt
    % stop inference at nans
    if any(isnan(Y(:, t)))
        break
    end
    
    %% integration (prediction) step
    % determine step size for integration until current measurement
    nsteps = ceil(Dt(t) / dt);
    dtt = Dt(t) / nsteps;
    
    % simple Euler integration
    for it = 1:nsteps
        % ffun(x, w, fopt), x - state, w - noise
        dXS = [ ffun(XS(1:nx, :), XS(nx+(1:nx), :), fopt); ...
                zeros(nx + ny, 2 * nxa + 1)];
        
        % integrate (noise must be accounted for in dXS)
        XS = XS + dtt * dXS;
    end
    mXpred(:,t) = XS(1:nx, :) * wm;
    XSerr = bsxfun(@minus, XS(1:nx, :), mXpred(:,t));
    Ppred{t} = XSerr * Wc * XSerr';
    
    
    %% update step
    % gfun(x, v, gopt), x - state, v - noise
    YS = gfun(XS(1:nx, :), XS(2*nx + (1:ny), :), gopt);
    mYpred(:,t) = YS * wm;
    YSerr = bsxfun(@minus, YS, mYpred(:,t));
    Pypred{t} = YSerr * Wc * YSerr';
    Cy = XSerr * Wc * YSerr';
    K{t} = Cy / Pypred{t};   % implmements Cy * inv(Pypred{t})
    
    peY(:,t) = Y(:,t) - mYpred(:,t);
    peX(:,t) = K{t} * peY(:,t);
    
    mX(:,t) = mXpred(:,t) + peX(:,t);
    P{t} = Ppred{t} - K{t} * Cy';   % K * Cy' = K * Pypred{t} * K';
    
    
    %% prepare next integration step by computing sigma points
    % make sure that P{t} is symmetric (may not be due to numerical errors)
    P{t} = (P{t} + P{t}') / 2;
    
    if t < nt
        try
            LP = chol(P{t}, 'lower');
        catch err
            % due to numerical errors P{t} may not be positive definite
            if strcmp(err.identifier,'MATLAB:posdef')
                nposdeferr = nposdeferr + 1;
                
                % make it positive definite by setting eigenvalues <= 0 to
                % some small value
                [V, D] = eig(P{t});
                D = diag(D);
                D(D <= 0) = 1e-15;
                LP = chol( V * diag(D) * V', 'lower' );
            else
                rethrow(err)
            end
        end
        % the Cholesky decomposition preserves block diagonal structure
        % such that we don't need to recompute the Cholesky decompositions
        % of Q and R
        A(1:nx, 1:nx) = LP;
        
        XS = bsxfun(@plus, [mX(:,t); zeros(nx+ny,1)], sqrtc*[zeros(nxa,1) A -A]);
    end
    
    
    %% printing progress
    % only print if full per cents changed
    if verbose && diff(floor([t-1,t]/(nt/100)))>=1
        fprintf('%s%3d%%',deletePast,round(t/nt*100));
    end
end

if verbose
    fprintf('\n');  % end of printing progress
end

if nposdeferr > 0
    warning('UKF:posdeferr', 'numerical errors fixed at runtime: made P{t} positive definite at %d time points', nposdeferr)
end
