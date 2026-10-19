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
    v_hat = zeros(size(v, 1) * size(v, 2), size(v, 3));
    
    for iframe = 1:size(v, 3)
        v_t = v(:, :, iframe);
        v_hat(:, iframe) = v_t(:) / norm(v_t(:));
    end
    
    c = zeros(n_frames - 2, 1);
    for iframe = 2:n_frames - 1
        c(iframe-1) = acos(dot(v_hat(:, iframe-1), v_hat(:, iframe)));
    end
end