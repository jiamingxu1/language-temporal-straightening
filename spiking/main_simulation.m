%% Setup
clear 
close all
clc

seed = 3;
rng(seed)

% make sure you are in the correct path, you should only see 'main_simlation.m', 'geometry/' and 'statistics/' in the folder hierarchy
addpath("geometry")
addpath("statistics/")

%% Simulate hierarchy of neural populations

n_frames = 11;        % number of video frames
frame_duration = 0.2; % duration over with a single frame was shown (in seconds)
n_trials = 1;         % number of trials (1 trial is enough when using t
is_analytical = true; % estimate curvature based on analytical solutions or simulated spike counts

% create time series
T = (n_frames * frame_duration);  % duration (in seconds)
dt = 0.001;                       % simulation time steps (in seconds)
x = linspace(0, T, T / dt);       % dummy data points
bin_size = 0.1;                   % size of time bin (in seconds)

% parameters of fast population
n_neurons = 20;                  % number of neurons within the population
rho_f = 3;                       % variance of tuning curves
tau_f = linspace(0.001, 2, 100); % time scale of tuning curves (in seconds)

% parameters of shared gain of fast/slow population
rho_g = 0.1;           % variance of gain
tau_g = [0.002, 0.02]; % time scale of gain (in seconds)
q_g = 2;               % power law exponent of gain covariance function

%% Simulate neural responses

% loop over hierarchy of neural populations
c_slow_gain = zeros(1, numel(tau_f));
c_fast_gain = zeros(1, numel(tau_f));

for itau = 1:numel(tau_f)
    fprintf('Running simulation for tau_f value number %d/%d\n\n', itau, numel(tau_f))
    
    % determine the range of the firing rate, can be used to manually force realistic ranges for the firing rate but increases runtime of the simulation, e.g., [1, 100] Hz
    mean_frange = []; 

    % simulate tuning curves
    [tuning_curves_fast_gain, ~, lambda_fast_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(1), q_g, mean_frange);
    [tuning_curves_slow_gain, ~, lambda_slow_gain] = compute_CMP(n_trials, x, n_neurons, rho_f, tau_f(itau), rho_g, tau_g(2), q_g, mean_frange);

    % get analytical mean and variance
    [mean_lambda_fast_gain, ~, ~, var_gain_fast_gain] = mean_var_CMP(tuning_curves_fast_gain, rho_g, dt, T, bin_size, tau_g(1), q_g);
    [mean_lambda_slow_gain, ~, ~, var_gain_slow_gain] = mean_var_CMP(tuning_curves_slow_gain, rho_g, dt, T, bin_size, tau_g(2), q_g);

    % compute global curvature (in degrees)
    y_fast_gain = compute_VST(var_gain_fast_gain, mean_lambda_fast_gain);
    y_slow_gain = compute_VST(var_gain_slow_gain, mean_lambda_slow_gain);
    c_fast_gain(itau) = rad2deg(mean(compute_curvature(y_fast_gain)));
    c_slow_gain(itau) = rad2deg(mean(compute_curvature(y_slow_gain)));
end

%% Plot results
figure(1)
set(gca, 'FontSize', 13)
scatter(tau_f, c_slow_gain, 10, 'filled')
hold on;
scatter(tau_f, c_fast_gain, 10, 'filled')
title('Neural trajectory curvature as a function of neural timescale')
xlabel('Time scale of f(s,t) (s)')
ylabel('Curvature (deg)')
legend('Slow gain', 'Fast gain')