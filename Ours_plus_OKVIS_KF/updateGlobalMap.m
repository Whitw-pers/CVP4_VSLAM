function worldPointSet = updateGlobalMap(worldPointSet, keyframeSet, ...
        optKeyframeSet)
% update world points after pose graph optimization
% INPUTS
% OUTPUTS

    oldPoses = keyframeSet.Views.AbsolutePose;
    newPoses = optKeyframeSet. Views.AbsolutePose;
    oldPoints = worldPointSet.WorldPoints;
    newPoints = oldPoints;

    % update loc of world points using new abs pose of corresponding major
    % view
    for i = 1:worldPointSet.Count

        majorViewIDs = worldPointSet.RepresentativeViewId(i);
        newPose = newPoses(majorViewIDs).A;
        tform = affinetform3d(newPose/oldPoses(majorViewIDs).A);
        newPoints(i, :) = transformPointsForward(tform, oldPoints(i, :)); 

    end

    worldPointSet = updateWorldPoints(worldPointSet, 1:worldPointSet.Count, newPoints);

end

