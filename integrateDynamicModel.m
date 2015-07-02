% [Y, X] = integrateDynamicModel(ffun, gfun, fopt, gopt, x0, T, options)
%
% integrates a nonlinear dynamic model which supports stochasticity, but
% noise is not used during integration (revealing the pure deterministic 
% dynamics of the model), uses simple Euler integration with (roughly)
% fixed step size
%
% in:
%       ffun    -   model dynamics function of the form
%                       ffun(x, w, fopt)
%                   where x is the current state vector and w is a vector
%                   of random perturbations which is set to 0 here
%       gfun    -   model observation function of the form
%                       gfun(x, v, gopt)
%                   where x is the current state vector and v is a vector
%                   of random perturbations which is set to 0 here
%       fopt    -   options used inside ffun
%       gopt    -   options used inside gfun, must have field:
%             .ny - dimensionality of observations (number of elements in
%                   the produced observation vectors)
%       x0      -   inital state vector
%                   [nx, 1] = size
%       T       -   times at which an observation should be produced
%                   counting from 0 for x0, i.e., if T(1)=0, Y(:,1) =
%                   gfun(x0, 0, gopt), must be monotonically increasing
%                   [1, nt] = size
%       options -   options structure with fields
%             .dt - desired size of integration step, determines how many
%                   Euler steps are made between two consecutive times in T
%                   note that the actually used dt at each step may be
%                   slightly different, this is done such that you always
%                   reach the times in T
% out:
%       Y       -   time series of observations resulting from model
%                   [ny, nt] = size
%       X       -   time series of hidden states resulting from model
%                   [nx, nt] = size
% author:
%       Copyright (C) 2015 Sebastian Bitzer
function [Y, X] = integrateDynamicModel(ffun, gfun, fopt, gopt, x0, T, options)

nx = numel(x0);
ny = gopt.ny;

if isnonemptyfield(options, 'dt')
    dt = options.dt;
else
    dt = 1;
end

nt = numel(T);
Dt = diff([0,T]);

x = x0;
w = zeros(nx, 1);
v = zeros(ny, 1);
X = nan(nx, nt);
Y = nan(ny, nt);
for t = 1:nt
    nsteps = ceil(Dt(t) / dt);
    dtt = Dt(t) / nsteps;

    % simple Euler integration
    for it = 1:nsteps
        % ffun(x, w, fopt), x - state, w - noise
        dx = ffun(x, w, fopt);            

        % integrate
        x = x + dtt * dx;
    end
    X(:, t) = x;
    Y(:, t) = gfun(x, v, gopt);
end
