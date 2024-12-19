function [worldPointSet,keyframeSet] = addKeyframe(worldPointSet, keyframeSet, ...
        currPose, currFeatures, currPoints, worldPointsIdx, featureIdx, ...
        localKeyframeIDs)
    %   Inputs:
%       worldPointSet - A structure containing the map of 3D world points.
%       keyframeSet - A structure containing the set of keyframes with
%                     associated camera poses and feature points.
%       currPose - The current pose of the camera, specified as a structure
%                  with fields for rotation (R) and translation vectors.
%       currFeatures - The current detected features in the new keyframe,
%                      typically stored as a structure with feature
%                      descriptors.
%       currPoints - The 2D image points corresponding to currFeatures.
%       worldPointsIdx - Indices of currFeatures that correspond to points in
%                        worldPointSet.
%       featureIdx - Indices mapping currFeatures to specific features in
%                    the existing keyframeSet.
%       localKeyframeIDs - Array of IDs of local keyframes to which the new
%                          keyframe should attempt to establish connections.
%
%   Outputs:
%       worldPointSet - Updated set of 3D world points.
%       keyframeSet - Updated set of keyframes with the new keyframe added
%                     and connections updated.
%
%   This function first assigns a new view ID to the incoming keyframe and
%   adds it to the keyframeSet. It then iterates through each local keyframe
%   specified by localKeyframeIDs to establish potential connections based
%   on shared visibility of world points. For each connection, it computes
%   a relative pose between the keyframes and checks for a sufficient number
%   of feature matches before adding the connection. The function also
%   updates correspondences in the worldPointSet with any new matches
%   found between current features and existing world points.
%
    viewID = keyframeSet.Views.ViewId(end)+1;

    keyframeSet = addView(keyframeSet,viewID, currPose,'Features',...
        currFeatures.Features,'Points',currPoints);

    keyFramePoses = keyframeSet.Views.AbsolutePose;
    
    

    for i = 1:numel(localKeyframeIDs)
        keyFrameID = localKeyframeIDs(i);
        
        if isSimMode()
            [idx3D,idx2D] = findWorldPointsInView(worldPointSet,keyFrameID);
        else
            [idx3DGenerated, idx2DGenerated] = findWorldPointsInView(worldPointSet,keyFrameID);
            idx3D = idx3DGenerated{1};
            idx2D = idx2DGenerated{1};
        end
        
        [~,index_past,index_current] = intersect(idx3D,worldPointsIdx,'stable');

        posePast = keyFramePoses(keyFrameID);

        poseRelative = rigidtform3d(posePast.R' * currPose.R, ...
            (currPose.Translation - posePast.Translation) * posePast.R);

        if numel(index_current) >= 6
            if isSimMode()
                keyframeSet = addConnection(keyframeSet,keyFrameID,viewID,...
                    poseRelative,'Matches', [idx2D(index_past),featureIdx(index_current)]);
            else
                coder.varsize('matches',[inf 2],[1,0]);
                matching_indices = [idx2D(index_past), featureIdx(index_current(:))];
                keyframeSet = addConnection(keyframeSet,keyFrameID,viewID,...
                    poseRelative,'Matches',matching_indices);
            end
        end
    end

    worldPointSet = addCorrespondences(worldPointSet, viewID,worldPointsIdx,featureIdx);

end

function tf = isSimMode()
    tf = isempty(coder.target);
end
