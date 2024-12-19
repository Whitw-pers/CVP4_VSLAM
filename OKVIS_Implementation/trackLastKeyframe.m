function [currPose, worldPointIdx, featureIdx] = trackLastKeyframe(...
    tracker, currImg, worldPoints, views, currFeatures, currPoints, ...
    lastKeyframeID, intrinsics)

    % match features from prev keyframe with known 3D world points
    [idx3D, idx2D] = findWorldPointsInView(worldPoints, lastKeyframeID);
    if ~isempty(coder.target())
        idx3D = idx3D{1};
        idx2D = idx2D{1};
    end
    lastKeyframeFeatures = views.Features{lastKeyframeID}(idx2D, :);

    [centerPoints, validity] = tracker(currImg);

    searchRadius = 4;

    matches = matchFeaturesInRadius(...
        binaryFeatures(lastKeyframeFeatures(validity, :)), ...
        binaryFeatures(currFeatures.Features), currPoints, ...
        centerPoints(validity, :), searchRadius, ...
        "MatchThreshold", 10, "MaxRatio", 0.9, "Unique", true);

    if size(matches, 1) < 20
        matches = matchFeaturesInRadius(...
            binaryFeatures(lastKeyframeFeatures(validity, :)), ...
            binaryFeatures(currFeatures.Features), currPoints, ...
            centerPoints(validity, :), 2 * searchRadius, ...
            "MatchThreshold", 10, "MaxRatio", 0.9, "Unique", true);
    end

    % tracking assumed lost
    if size(matches, 1) < 10
        pose = rigidtform3d;
        currPose = repmat(pose, 0, 0);
        worldPointIdx = zeros(0, 1);
        featureIdx = zeros(0, 1, class(matches));
        return
    end

    % get index of matched 3D world points and 2D features
    tempIdx = find(validity); % Convert to linear index
    coder.varsize('worldPointIdx', [inf, 1], [1, 0]);
    worldPointIdx = idx3D(tempIdx(matches(:,1)));
    coder.varsize('featureIdx', [inf, 1], [1, 0]);
    featureIdx = matches(:,2);

    matches2D = currPoints.Location(featureIdx,:);
    matches3D = worldPoints.WorldPoints(worldPointIdx, :);

    % estimate camera pose with PNP
    matches2D = cast(matches2D, 'like', matches3D);
    [currPose, inliers] = estworldpose(matches2D, matches3D, intrinsics, ...
        'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);

    % refine camera pose with bundle adjustment
    currPose = bundleAdjustmentMotion(matches3D(inliers,:), ... 
        matches2D(inliers,:), currPose, intrinsics, ...
        'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7, ...
        'RelativeTolerance', 1e-15, 'MaxIterations', 20);

    worldPointIdx = worldPointIdx(inliers);
    featureIdx = featureIdx(inliers);
end

