% Compute curvature betweeen rate vectors in discriminability space, based
% on Henaff et al. (2021).
%
% Inputs:
%    y - (n_neurons x n_bins) Rate vectors after the variance-stabilizing transformation (VST)
%
% Outputs:
%   c - (1 x n_bins - 2) Curvature of neural embeddings 

function c = compute_curvature(y)
    c = compute_trajectory_pixel(reshape(y, [1, size(y, 1), size(y, 2)]));
end