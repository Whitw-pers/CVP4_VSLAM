function [isLoopClosed, worldPointSet, keyFrameSet] = addLoopConnection(...
    worldPointSet, keyFrameSet, loopCandidates, currentKeyFrameId, ...
    currentFrameFeatures)


loopClosureConnections = zeros(0, 2, 'uint32');
requiredNumMatches = 50; % Define a threshold for the number of matches required for a loop closure

numCandidates = size(loopCandidates, 1);
if isSimulationMode()
    [worldPointIndicesCurrent, featureIndicesCurrent] = findWorldPointsInView(worldPointSet, currentKeyFrameId);
else
    [worldPointIndicesCurrentCg, featureIndicesCurrentCg] = findWorldPointsInView(worldPointSet, currentKeyFrameId);
    featureIndicesCurrent = featureIndicesCurrentCg{1};
    worldPointIndicesCurrent = worldPointIndicesCurrentCg{1};
end

currentValidFeatures = currentFrameFeatures.Features(featureIndicesCurrent, :);

for k = 1:numCandidates
    if isSimulationMode()
        [worldPointIndicesCandidate, featureIndicesCandidate] = findWorldPointsInView(worldPointSet, loopCandidates(k));
    else
        [worldPointIndicesCandidateCg, featureIndicesCandidateCg] = findWorldPointsInView(worldPointSet, loopCandidates(k));
        featureIndicesCandidate = featureIndicesCandidateCg{1};
        worldPointIndicesCandidate = worldPointIndicesCandidateCg{1};
    end
    candidateFrameFeatures = keyFrameSet.Views.Features{loopCandidates(k)};

    candidateValidFeatures = candidateFrameFeatures(featureIndicesCandidate, :);

    featureIndexPairs = matchFeatures(binaryFeatures(currentValidFeatures), binaryFeatures(candidateValidFeatures), ...
        'Unique', true, 'MaxRatio', 0.9, 'MatchThreshold', 40);

    if size(featureIndexPairs, 1) < requiredNumMatches
        continue
    end

    [worldPointsInCurrentCamera, worldPointsInCandidateCamera] = transformWorldPointsToCamera(...
        worldPointSet, keyFrameSet, currentKeyFrameId, loopCandidates(k), featureIndexPairs, ...
        worldPointIndicesCurrent, worldPointIndicesCandidate);

    warningState = warning('off','all');
    if isSimulationMode()
        [relativePose, inlierIndices] = estimateGeometricTransform3D(...
            worldPointsInCurrentCamera, worldPointsInCandidateCamera, 'similarity', 'MaxDistance', 0.1);
    else
        [relativePose, inlierIndices] = estimateGeometricTransform3D(...
            worldPointsInCurrentCamera, worldPointsInCandidateCamera, 'rigid', 'MaxDistance', 0.1);
    end
    warning(warningState);

    inlierFeatureIndices = inlierIndices(:);
    matchedFeatureIndices1 = featureIndexPairs(inlierFeatureIndices, 1);
    matchedFeatureIndices2 = featureIndexPairs(inlierFeatureIndices, 2);
    featureIndicesCandidateMatched = featureIndicesCandidate(matchedFeatureIndices2);
    featureIndicesCurrentMatched = featureIndicesCurrent(matchedFeatureIndices1);
    matchIndices = uint32([featureIndicesCandidateMatched, featureIndicesCurrentMatched]);
    keyFrameSet = addConnection(keyFrameSet, loopCandidates(k), currentKeyFrameId, relativePose, 'Matches', matchIndices);
    if isSimulationMode()
        disp(['Loop edge added between keyframe: ', num2str(loopCandidates(k)), ' and ', num2str(currentKeyFrameId)]);
    end

    matchedWorldPointIndices1 = worldPointIndicesCurrent(matchedFeatureIndices1);
    matchedWorldPointIndices2 = worldPointIndicesCandidate(matchedFeatureIndices2);

    worldPointSet = updateWorldPoints(worldPointSet, matchedWorldPointIndices1, worldPointSet.WorldPoints(matchedWorldPointIndices2, :));

    loopClosureConnections = [loopClosureConnections; loopCandidates(k), currentKeyFrameId];
end
isLoopClosed = ~isempty(loopClosureConnections);
end

function tf = isSimulationMode()
tf = isempty(coder.target);
end

function [pointsInCurrentCamera, pointsInCandidateCamera] = transformWorldPointsToCamera(...
    worldPointSet, keyFrameSet, currentKeyFrameId, candidateKeyFrameId, featureIndexPairs, ...
    worldPointIndicesCurrent, worldPointIndicesCandidate)
% Transform world points to camera frame

% Retrieve world points for current and candidate frames using matched feature indices
currentWorldPoints = worldPointSet.WorldPoints(worldPointIndicesCurrent(featureIndexPairs(:, 1)), :);
candidateWorldPoints = worldPointSet.WorldPoints(worldPointIndicesCandidate(featureIndexPairs(:, 2)), :);

% Get the pose transformations for the current and candidate keyframes
currentPoseTransform = poseToExtrinsics(keyFrameSet.Views.AbsolutePose(currentKeyFrameId));
candidatePoseTransform = poseToExtrinsics(keyFrameSet.Views.AbsolutePose(candidateKeyFrameId));

% Transform world points to their respective camera coordinates
pointsInCurrentCamera = transformPointsForward(currentPoseTransform, currentWorldPoints);
pointsInCandidateCamera = transformPointsForward(candidatePoseTransform, candidateWorldPoints);
end
