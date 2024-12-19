function [F, inliers] = getFundamentalMatrix(matches1, matches2, intrinsics)
% gets Fundamental matrix from matches between images and camera intrinsics
% if algorithm utilizes both fundamental matrix and homography, will need
% to add computation of heuristic score
% INPUTS
    % matches1: matched features from source image
    % matches2: matched features from target image
    % intrinsics: cameraIntrinsics object containing camera intrinsic
    % matrix K
% OUTPUTS
    % F: Fundamental matrix
    % inliers: indexes of inlying points used to compute essential matrix

% just like in SfM, get essential matrix first
% MaxNumTrials increased from default settings
% MaxDistance (Sampson distance threshold) increased to reduce convergence time
% can play with these settings
[E, inliersLogicalIndex] = estimateEssentialMatrix(matches1, matches2, intrinsics, MaxNumTrials=1e3, MaxDistance=4);

% get F from E
F = intrinsics.K' \ E / intrinsics.K;

% inliers1 = matches1(inliersLogicalIndex);
% inliers2 = matches2(inliersLogicalIndex);

inliers  = find(inliersLogicalIndex);

end

