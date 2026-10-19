% Radial basis function (RBF) kernel.
%
% Inputs: 
%   t1  - (1 x T) Time series
%   t2  - (1 x T) Time series
%   rho - [double] Variance
%   tau - [1 x n_diverse_neurons] Length scale
%
% Output:
%   K - (T x T x n_diverse_neurons) Covariance matrix, n_diverse_neurons refers to the number of neurons with different timescales

function K = rbf_kernel(t1, t2, rho, tau)
    D2 = (t1(:) - t2(:)').^2;
    K = rho * exp(-D2 ./ (2 * reshape(tau.^2, 1, 1, [])));
end