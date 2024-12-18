function [isDetected, loopKeyframeIDs] = checkLoopClosure(keyframeSet, ...
            currKeyframeID, loopDatabase, currFeatures)
% INPUTS
% OUTPUTS

    % get visually similar keyframes
    [candidateViewIDs, similarityScore] = detectLoop(loopDatabase, currFeatures);
    candidateViewIDs = double(candidateViewIDs');
    similarityScore = double(similarityScore');

    % find similarity between current keyframe and strongly connected
    % keyframes
    % min similarity score used as baseline to refine candidate keyframes
    % for loop closure, which are visually similar but not connected
    connViews = connectedViews(keyframeSet, currKeyframeID);
    connViewIDs = connViews.ViewId;
    strongConnViews = connectedViews(keyframeSet, currKeyframeID, MinNumMatches = 50);
    strongConnViewIDs = strongConnViews.ViewId;

    % get 10 most similar connected keyframes
    [~, ~, ib] = intersect(strongConnViewIDs, candidateViewIDs);
    minScore = min(similarityScore(ib));

    [loopKeyframeIDs, ia] = setdiff(candidateViewIDs, connViewIDs, 'stable');

    % scores of non connected keyframes
    candidateScores = similarityScore(ia);

    if ~isempty(ia) && ~isempty(ib)

        bestScore = candidateScores(1);

        % to be valid, score must be greater than 0.75*bestScore and
        % minScore
        isValid = candidateScores > max(0.75*bestScore, minScore);
        loopKeyframeIDs = loopKeyframeIDs(isValid);

    else

        loopKeyframeIDs = zeros(coder.ignoreConst(0), coder.ignoreConst(0), class(loopKeyframeIDs));

    end

    % loop detection criteria:
    % at least 3 consecutive candidate keyframes found
    if size(loopKeyframeIDs, 1) >= 3

        groups = nchoosek(loopKeyframeIDs, 3);
        consecGroups = groups(max(groups,[],2) - min(groups,[],2) < 4, :);
        if ~isEmpty(consecGroups)

            loopKeyframeIDs = consecGroups(1, :);
            isDetected = true;

        else

            isDetected = false;
            
        end

    else

        isDetected = false;

    end

end

