function [localKeyframeIDs, currPose, worldPointsIdx, featureIdx, isKeyframe] = ...
            stockKeyframeSelector(worldPointSet, keyframeSet, worldPointsIdx, ...
            featureIdx, currPose, currFeatures, currPoints, intrinsics, ...
            newKeyframeAdded, lastKeyframe, currFrame)
% performs role of helperTrackLocalMap()
% INPUTS:
    % worldPointSet:
    % keyframeSet:
    % worldPointsIdx:
    % featureIdx:
    % currPose:
    % currFeatures:
    % currPoints:
    % intrinsics:
    % newKeyframeAdded:
    % lastKeyframe:
    % currFrame:
% OUTPUTS:
    % localKeyframeIDs:
    % currPose:
    % worldPointsIdx:
    % featureIdx:
    % isKeyframe:

    % adjust parameters to tune performance of stock keyframe selector
    numSkipFrames = 20;
    numPointsKeyframe = 90;
    scaleFactor = 1.2;

    persistent numPointsRefKeyframe localPointsIdx localKeyframeIDsInternal

    if isempty(numPointsRefKeyframe) || newKeyframeAdded
        [localPointsIdx, localKeyframeIDsInternal, numPointsRefKeyframe] = ...
            updateRefKeyframeAndLocalPoints(worldPointSet, keyframeSet, worldPointsIdx);
    end

    % project world points into frame and search for more 2D-3D point
    % correspondences
    newWorldPointIdx = setdiff(localPointsIdx, worldPointsIdx, 'stable');
    [localFeatures, localPoints] = getFeatures(worldPointSet, keyframeSet.Views, ...
        newWorldPointIdx);
    [projectedPoints, inliersIdx] = removeOutlierWorldPoints(worldPointSet, ...
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

    isKeyframe = checkKeyframe(numPointsRefKeyframe, lastKeyframe, currFrame, ...
        worldPointsIdx, numSkipFrames, numPointsKeyframe);
    localKeyframeIDs = localKeyframeIDsInternal;

    if isKeyframe
        
        currPose = bundleAdjustmentMotion(matchedWorldPoints, matchedImagePoints, ...
            currPose, intrinsics, 'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7, ...
            'RelativeTolerance', 1e-16,'MaxIteration', 20);

    end

end

%% internal functions called by stockKeyframeSelector()
function [localPointsIdx, localKeyframeIDs, numPointsRefKeyframe] = ...
    updateRefKeyframeAndLocalPoints(worldPoints, keyframeSet, pointsIdxs)
    
    if keyframeSet.NumViews == 1
        localKeyframeIDs = keyframeSet.Views.ViewId;
        localPointsIdx = (1:worldPoints.Count)';
        numPointsRefKeyframe = worldPoints.Count;
        return
    end

    % the reference keyframe has the most covisible 3D world points
    viewIDs = findViewsOfWorldPoint(worldPoints, pointsIdxs);
    refKeyframeID = mode(vertcat(viewIDs{:}));

    localKeyframes = connectedViews(keyframeSet, refKeyframeID, "MaxDistance", 2);
    localKeyframeIDs = [localKeyframes.ViewId; refKeyframeID];

    pointIdx = findWorldPointsInView(worldPoints, localKeyframeIDs);
    if iscell(pointIdx)
        numPointsRefKeyframe = numel(pointIdx{localKeyframeIDs == refKeyframeID});
        localPointsIdx = sort(vertcat(pointIdx{:}));
    else
        numPointsRefKeyframe = numel(pointIdx);
        localPointsIdx = sort(pointIdx);
    end
end

function [features, points] = getFeatures(worldPoints, views, worldPointIdx)

    allIdxs = zeros(1, numel(worldPoints));

    count = [];
    viewFeatures = views.Features;
    viewPoints = views.Points;

    for i = 1:numel(worldPointIdx)
        idx3D = worldPointIdx(i);
        viewID = double(worldPoints.RepresentativeViewId(idx3D));

        if isempty(count)
            count = [viewID, size(viewFeatures{viewID}, 1)];
        elseif ~any(count(:, 1) == viewID)
            count = [count; viewID, size(viewFeatures{viewID}, 1)];
        end

        idx = find(count(:, 1) == viewID);
        if idx > 1
            offset = sum(count(1:idx - 1, 2));
        else
            offset = 0;
        end
        allIdxs(i) = worldPoints.RepresentativeFeatureIndex(idx3D) + offset;

    end

    uIDs = count(:, 1);

    % concat features and indexes is faster than processing via for loop
    allFeatures = vertcat(viewFeatures{uIDs});
    features = allFeatures(allIdxs, :);
    allPoints = vertcat(viewPoints{uIDs});
    points = allPoints(allIdxs, :);

end

function [projectedPoints, inliers] = removeOutlierWorldPoints(worldPoints, ...
    pose, intrinsics, localPointsIdx, scaleFactor)
    
    % points within image bounds
    points3D = worldPoints.WorldPoints(localPointsIdx, :);
    [projectedPoints, isInImage] = world2img(points3D, pose2extr(pose), intrinsics);

    if isempty(projectedPoints)
        error('Tracking failed. Try inserting new key frames more frequently.')
    end

    % parallax < 60 degrees
    cam2points = points3D - pose.Translation;
    viewDirection = worldPoints.ViewingDirection(localPointsIdx, :);
    validByView = sum(viewDirection .* cam2points, 2) > cosd(60)*(vecnorm(cam2points, 2, 2));

    % distance from world point to camera center is in range of scale
    % invariant depth
    minDist = worldPoints.DistanceLimits(localPointsIdx, 1)/scaleFactor;
    maxDist = worldPoints.DistanceLimits(localPointsIdx, 2)*scaleFactor;
    dist = vecnorm(points3D - pose.Translation, 2, 2);
    validByDistance = dist > minDist & dist < maxDist;
    
    inliers = isInImage & validByView & validByDistance;

    projectedPoints = projectedPoints(inliers, :);

end

function isKeyframe = checkKeyframe(numPointsRefKeyframe, lastKeyframe, currFrame, ...
        worldPointsIdx, numSkipFrames, numPointsKeyframe)
    % insert new keyframe if:

    % more than numSkipFrames passed since last keyframe
    tooManySkipFrames = currFrame > lastKeyframe + numSkipFrames;
    
    % track less than numPointsKeyrame world points
    tooFewWorldPoints = numel(worldPointsIdx) < numPointsKeyframe;

    % tracked world points are less than 90% of points tracked by the
    % reference keyframe
    tooFewTrackedPoints = numel(worldPointsIdx) < 0.9 * numPointsRefKeyframe;

    isKeyframe = (tooManySkipFrames || tooFewWorldPoints) && tooFewTrackedPoints;

end