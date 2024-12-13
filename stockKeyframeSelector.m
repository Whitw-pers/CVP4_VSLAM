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

    persistent numPointsRefKeyframe localPointsIdx localKeyframeIDsInternal

    if isempty(numPointsRefKeyframe) || newKeyframeAdded
        [localPointsIdx, localKeyframeIDsInternal, numPointsRefKeyframe] = ...
            updateRefKeyFrameAndLocalPoints(worldPointSet, keyframeSet, worldPointsIdx);
    end

    

end
