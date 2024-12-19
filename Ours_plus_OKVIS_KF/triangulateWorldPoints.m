function [worldPoints, inliers, isValid] = triangulateWorldPoints(pose1, pose2, matches1, matches2, intrinsics)
% trangulates 3D world points and checks validity of solution
% INPUTS
    % pose1: pose of camera at first frame
    % pose2: pose of next frame
    % matches1: matched points from first frame
    % matches2: matched points from second frame
    % intrinsics: cameraIntrinsics object
% OUTPUTS
    % worldPoints: 3D world points
    % inliers: points that pass view direction and reprojection error test
    % isValid: check of sufficient parallax between views

% find projection matrix for each camera with camera intrinsics and
% extrinsics
camProj1 = cameraProjection(intrinsics, pose2extr(pose1));
camProj2 = cameraProjection(intrinsics, pose2extr(pose2));

% triangulate 3D world points
[worldPoints, reprojErr, isInFront] = triangulate(matches1, matches2, camProj1, camProj2);

% test points by view direction and reprojection error
inliers = isInFront & reprojErr < 1;
worldPoints = worldPoints(inliers, :);

% check validity by confirming there is significant parallax between the
% views
minParallax = 1;    % in degrees
ray1 = worldPoints - pose1.Translation;
ray2 = worldPoints - pose2.Translation;
cosAngle = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));
isValid = all(cosAngle < cosd(minParallax) & cosAngle>0);
end

