% this script demonstrate the use of the UKF function

%% define nonlinear model
% setup Hopfield network
nx = 2;

k = 100;
g = 10;
blat = 1.7;
blin = blat / 2 / g;
L = blat * (eye(nx) - ones(nx));
o = g;
r = 1;

sigx = @(X) 1 ./ (1 + exp( -r*(X - o) ) );
dHop = @(X) k * (L*sigx(X) + blin*(g - X));

% define observation function
pos = [-1 1; -2 2; -3 3];
nd = size(pos, 1);

sigobs = @(X) 1 ./ ( 1 + exp( -0.7 * (X - g / 2) ) );

obsf = @(X) pos * sigobs(X);


%% generate data
gopt.ny = nd;

% note that the dynamics and observation functions used with the UKF need
% to define how noise enters the computations: here we simply add the noise
ffun = @(X, W, fopt) dHop(X) + W;
gfun = @(X, V, gopt) obsf(X) + V;

x0 = rand(2, 1) * g;

nt = 200;
T = linspace(0, 1.5, nt);

intopt.dt = 0.001;

% initial, standard dynamics of Hopfield network
[Y(:, 1:nt/2), X(:, 1:nt/2)] = integrateDynamicModel(ffun, gfun, [], ...
    gopt, x0, T(1:nt/2), intopt);
% simulate a switch to another attractor by restarting the dynamics in the
% same initial state, but with state variables exchanged
[Y(:, nt/2+1:nt), X(:, nt/2+1:nt)] = integrateDynamicModel(ffun, gfun, [], ...
    gopt, x0(2:-1:1), T(1:nt/2), intopt);

figure(249)
clf
vis.realx = plot(T, X', 'LineWidth', 2);
xlabel('time (s)')
ylabel('Hopfield states')
legend('x_1', 'x_2')


% add some noise to observations
noisestd = 1;
Ynoise = Y + randn( size(Y) ) * noisestd;

figure(250)
clf
plot(T, Y', '--')
hold on
plot(T, Ynoise')
xlabel('time (s)')
ylabel('observations')


%% filter with UKF
x0ukf = rand(nx, 1) * g;

% arbitrary prior uncertainty over initial state
P0 = eye(nx);

% you need a really high uncertainty on the Hopfield states to be able to
% capture the switch into the other attractor
Q = eye(nx) * 50;
% use actual size of noise in model
R = eye(nd) * noisestd;

ukfopt = struct([]);

[mX, P, peY, peX, mYpred, mXpred, Ppred, Pypred, nposdeferr] = ...
    UKF(Ynoise, T, x0ukf, P0, ffun, gfun, Q, R, ukfopt);


figure(249)
hold on
vis.ukfx = plot(T, mX', '--');
legend([vis.realx; vis.realx(1); vis.ukfx(1)], 'x_1', 'x_2', 'real', 'ukf', ...
    'Location', 'E')
