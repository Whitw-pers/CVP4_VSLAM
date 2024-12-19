function [features, validPoints] = extractORBFeatures(RGBimg, numPoints)
% gets ORB features and descriptors
% consider adding scale factor and numLevels if performance is poor
% INPUTS
    % RGBimg:       RGB image to get features from
    % numPoints:    number of uniformly distributed points to select
% OUTPUTS
    % features:     descriptors for each feature extracted from RGBimg
    % validPoints:  xy coords of points associated with feature descriptors

% detect ORB features
grayImg  = im2gray(RGBimg);
points = detectORBFeatures(grayImg);

% Select a subset of features, uniformly distributed throughout the image
points = selectUniform(points, numPoints, size(grayImg, 1:2));

% Extract features
[features, validPoints] = extractFeatures(grayImg, points);
end

