% Compute variance-stabilizing transformation.

% Inputs:
%   var_gain    - (1 x n_bins) Variance of the gain
%   mean_lambda - (n_neurons x n_bins) Mean of the modulated rate function
%
% Output:
%   y - (n_neurons x n_bins) Embedding after variance-stabilizing transformation

function y = compute_VST(var_gain, mean_lambda)
       y = 2 ./ sqrt(var_gain) .* asinh(sqrt(var_gain) .* sqrt(mean_lambda)); % this embedding was derived for constant gain
end