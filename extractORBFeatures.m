function [features, validPoints] = extractORBFeatures(RGBimg, scaleFactor, numLevels, numPoints)
% gets ORB features and descriptors
% INPUTS
    % RGBimg:
    % scaleFactor:
    % numLevels:
    % numPoints:
% OUTPUTS
    % features:
    % validPoints:

% detect ORB features
grayImg  = im2gray(RGBimg);
points = detectORBFeatures(grayImg, ScaleFactor=scaleFactor, NumLevels=numLevels);

% Select a subset of features, uniformly distributed throughout the image
points = selectUniform(points, numPoints, size(grayImg, 1:2));

% Extract features
[features, validPoints] = extractFeatures(grayImg, points);
end

