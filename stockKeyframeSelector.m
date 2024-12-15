function [localKeyframeIDs, currPose, worldPointsIdx, featureIdx, isKeyframe] = ...
            stockKeyframeSelector(worldPointSet, keyframeSet, worldPointsIdx, ...
            featureIdx, currPose, currFeatures, currPoints, intrinsics, ...
            isLastKeyframe, lastKeyframe, currFrame);
% performs role of helperTrackLocalMap()
% INPUTS:
% OUTPUTS:

    % adjust parameters to tune performance of stock keyframe selector
    numSkipFrames = 20;
    numPointsKeyframe = 90;
    scaleFactor = 1.2;

    persistent numPointsRefKeyframe localPointsIdx localKeyframeIDsInternal

    if isempty(numPointsRefKeyframe) || newKeyframeAdded
        [localPointsIdx, localKeyframeIDsInternal, numPointsRefKeyframe] = ...
            updateRefKeyFrameAndLocalPoints(worldPointSet, keyframeSet, worldPointsIdx);
    end

    % project world points into frame and search for more 2D-3D point
    % correspondences
    newWorldPointIdx = setdiff(localPointsIdx, worldPointsIdx, 'stable');
    [localFeatures, localPoints] = getFeatures(worldPointSet, keyframeSet.Views, ...
        newWorldPointIdx);
    [projectedPoints, inliersIdx] = removeOutlierMapPoints(worldPointSet, ...
        currPose, intrinsics, newWorldPointIdx, scaleFactor);

    newWorldPointIdx = newWorldPointIdx(inliersIdx);
    localFeatures = localFeatures(inliersIdx, :);
    localPoints = localPoints(inliersIdx);

    unmatchedFeatureIdx = setdiff(cast((1:size(currFeatures.Features, 1)).', 'uint32'), ...
        featureIdx, 'stable');
    unmatchedFeatures = currFeatures.Features(unmatchedFeatureIdx, :);
    unmatchedValidPoints = currPoints(unmatchedFeatureIdx);

    % define search radius based on scale and view direction
    radius = 4 * ones(size(localFeatures, 1), 1) * scaleFactor^2;

    matches = matchFeaturesInRadius(binaryFeatures(localFeatures), ...
        binaryFeatures(unmatchedFeatures), unmatchedValidPoints, projectedPoints, ...
        radius, 'MatchThreshold', 40, 'MaxRatio', 1, 'Unique', true);

    % filter matches by scale
    isGoodScale = unmatchedValidPoints.Scale(matches(:, 2)) >= localPoints.Scale(matches(:, 1))/scaleFactor^2 & ...
        unmatchedValidPoints.Scale(matches(:, 2)) <= localPoints.Scale(matches(:, 1))/scaleFactor^2;
    matches = matches(isGoodScale, :);

    % if currFrame is keyframe, refine pose with additional 2D-3D correspondences
    worldPointsIdx = [newWorldPointIdx(matches(:, 1)); worldPointsIdx];
    featureIdx = [unmatchedFeatureIdx(matches(:, 2)); featureIdx];
    matchedWorldPoints = worldPointSet.WorldPoints(worldPointsIdx, :);
    matchedImagePoints = currPoints.Location(featureIdx, :);

    isKeyframe = checkKeyFrame(numPointsRefKeyframe, lastKeyframe, currFrame, ...
        worldPointsIdx, numSkipFrames, numPointsKeyframe);
    localKeyframeIDs = localKeyframeIDsInternal;

    if isKeyframe
        
        currPose = bundleAdjustmentMotion(matchedWorldPoints, matchedImagePoints, ...
            currPose, intrinsics, PointsUndistorted', true, 'AbsoluteTolerance', 1e-7, ...
            'RelativeTolerance', 1e-16,'MaxIteration', 20);

    end

end

