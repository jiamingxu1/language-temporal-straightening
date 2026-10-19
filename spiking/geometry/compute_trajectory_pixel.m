% Compute pixel-wise curvature based on Hénaff et al. (2019).
%
% Inputs:
%   frame_mat - [n_pixels x n_pixels x n_frames] video array
%
% Outputs:
%   c        - [n_frames - 2] Estimated curvatures in rads

function c = compute_trajectory_pixel(frame_mat)

    n_frames = size(frame_mat, 3);
    
    % compute local trajectory
    v = diff(frame_mat, 1, 3);
    v_hat = reshape(v, [], n_frames-1) ./ vecnorm(reshape(v, [], n_frames-1), 2, 1);
    
    v_hat1 = v_hat(:, 1:n_frames-2);
    v_hat2 = v_hat(:, 2:n_frames-1);
    c = acos(dot(v_hat1, v_hat2));
end