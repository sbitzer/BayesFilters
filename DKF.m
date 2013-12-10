% [] = DKF(Y,x0,P0,A,B,C,Q,R,options)
%
% discrete Kalman filter, where the linear dynamical system is defined as
%
%    x_t+1 = Ax_t + b_t + w
%      y_t = Cx_t + v
% 
% where w and v are Gaussian noise variables with covariances Q and R,
% respectively, note that I have compacted the usual Bu_t = b_t
%
% note that the means and covariances stored in mYpred and Pypred define
% the marginal likelihood distribution p(y_t|Y_t-1)
function [mX, Pstore ,peY, peX, mYpred, mXpred, Ppred, Pypred] = ...
            DKF(Y, x0, P0, A, B, C, Q, R)

[ny,nt] = size(Y);
nx = numel(x0);

% secure memory
mX = nan(nx,nt);
peY = nan(ny,nt);
peX = nan(nx,nt);
mYpred = nan(ny,nt);
mXpred = nan(nx,nt);
Pstore = cell(1,nt);
Ppred = cell(1,nt);
Pypred = cell(1,nt);

% initialise the prior
mu = x0;
P = P0;

% run the filter
for t = 1:nt
    % predict
    mXpred(:,t) = A * mu + B(:,t);
    mYpred(:,t) = C * mXpred(:,t);
    Ppred{t} = A * P * A' + Q;
    Pypred{t} = C * Ppred{t} * C' + R;
    
    % update
    K = Ppred{t} * C' / Pypred{t};
    
    peY(:,t) = Y(:,t) - mYpred(:,t);
    peX(:,t) = K * peY(:,t);
    
    mX(:,t) = mXpred(:,t) + peX(:,t);
    Pstore{t} = (eye(nx) - K * C) * Ppred{t};
    
    % prior for next time = current posterior
    mu = mX(:,t);
    P = Pstore{t};
end

