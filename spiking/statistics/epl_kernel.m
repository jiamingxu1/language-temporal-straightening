% Exponentiated power law (EPL) kernel
%
% Inputs: 
%   t1  - (1 x T) Time series
%   t2  - (1 x T) Time series
%   rho - [double] Variance
%   tau - [double] Length scale
%   q   - [double] (0 < q <= 2) Exponent
%
% Output:
%   K - (T x T) Covariance matrix

function K = epl_kernel(t1, t2, rho, tau, q)
    K = rho * exp(-1/2 * abs((t1 - t2) / tau).^q);
end