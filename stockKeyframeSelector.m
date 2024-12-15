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

end

