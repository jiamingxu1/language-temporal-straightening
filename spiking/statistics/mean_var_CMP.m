% Compute mean and variance of spike counts under the CMP model.
%
% Inputs:
%   tuning_curves - (n_neurons x length(x) x n_trials) Array of simulated tuning curves
%   rho_g         - [double] Variance of covariance function of gain
%   dt            - [double] Simulation time steps (in s)
%   bin_size      - [double] Bin size (in seconds)
%   tau_g         - [double] Length scale of the gain
%   rho_g         - [double] Variance of gain
%   q             - [double] (0 < q <= 2) Exponent
%
% Outputs:
%   mean_spike_count - (n_neurons x n_bins x n_trials) Mean spike count per time bin
%   var_spike_count  - (n_neurons x n_bins x n_trials) Variance per time bin
%   mean_gain        - [double] Mean of gain
%   var_gain         - (n_neurons x n_bins) Variance of gain per time bin

function [mean_spike_count, var_spike_count, mean_gain, var_gain] = mean_var_CMP(tuning_curves, rho_g, dt, T, bin_size, tau_g, q_g)
    
    if nargin < 8
        n_trials = 1;
    end

    % tuning curves are equal across trials
    tuning_curves = squeeze(tuning_curves(:, :, 1)); 

     % downsampling
    bin_factor = bin_size / dt;
    tuning_curves_reshaped = reshape(tuning_curves, size(tuning_curves, 1), bin_factor, []);  % (n_neurons x bin_factor x n_bins)
    
    % compute mean of spike count (Methods under Partitioning variability with CMP)
    mean_spike_count = squeeze(exp(rho_g/2) .* sum(tuning_curves_reshaped, 2)) * dt;
    
    % get variance of lambda and gain
    [var_spike_count, var_gain] = compute_integrals(mean_spike_count, tuning_curves, tuning_curves_reshaped, bin_factor, T, tau_g, rho_g, q_g, dt, n_trials); 

    % compute mean of gain
    mean_gain = exp(1/2 * rho_g); % K_g is a stationary kernel that only depends on time lag (t1-t2) so doesn't depend on bin size; if time lag = 0, covariance collapses to rho_g
end

function [var_count, var_gain] = compute_integrals(mean_count, tuning_curves, tuning_curves_reshaped, bin_factor, T, tau_g, rho_g, q_g, dt, n_trials)
    
    time_total = linspace(0, T, size(tuning_curves, 2));
    time_bins = reshape(time_total, bin_factor, []);
    [n_neurons, bin_size, ~] = size(tuning_curves_reshaped);

    gamma = zeros(n_neurons, size(time_bins, 2));
    var_gain = zeros(n_neurons, size(time_bins, 2));
    % mean_gain = zeros(1, size(time_bins, 2));
    for ibin = 1:size(gamma, 2)

        % generate grid points
        [T1, T2] = meshgrid(time_bins(:, ibin), time_bins(:, ibin));

        K_g = exp(epl_kernel(T1, T2, rho_g, tau_g, q_g));

        % compute gamma for variance of lambda (gamma is the second summand in the variance equation)
        tuning_squared = reshape(tuning_curves_reshaped(:, :, ibin), [n_neurons, bin_size, 1]) .* reshape(tuning_curves_reshaped(:, :, ibin), [n_neurons, 1, bin_size]);
        gamma(:, ibin) = exp(rho_g) * dt^2 * sum(tuning_squared .* (permute(repmat(K_g, 1, 1, n_neurons), [3, 2, 1]) - 1), [2, 3]); 
        
        % % compute mean of gain
        % mean_gain(ibin, itrial) = sum(exp(1/2 * rho_g));

        % compute variance of gain
        var_gain(:, ibin) = (gamma(:, ibin) / exp(rho_g)) ./ (dt^2 * sum(reshape(tuning_curves_reshaped(:, :, ibin), [n_neurons, bin_size, 1]), 2).^2);
    end
    var_count = mean_count .* ones(size(mean_count, 1), size(mean_count, 2), n_trials) + var_gain;
end
