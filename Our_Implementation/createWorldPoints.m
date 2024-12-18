function [worldPointSet,keyframeSet, newPointIdx] = createWorldPoints(worldPointSet,...
    keyframeSet,currKeyframeID,intrinsics)
    scaleFactor = 1.2;
    min_Matches = 10;
    min_Parallax   = 3; %degrees

    linked_views = connectedViews(keyframeSet,currKeyframeID,'MinNumMatches',min_Matches);
    linked_IDs = linked_views.ViewId;

    currPose = keyframeSet.Views.AbsolutePose(currKeyframeID);
    currFeatures = keyframeSet.Views.Features{currKeyframeID};
    currPoints = keyframeSet.Views.Points{currKeyframeID};
    currLocations = currPoints.Location;
    currScales = currPoints.Scale;

    currProjMatrix = cameraProjection(intrinsics,pose2extr(currPose));

    newPointIdx  = zeros(0, 1);
    for i = 1:numel(linked_IDs)
        iteration_pose = keyframeSet.Views.AbsolutePose(linked_IDs(i));
        if isSimMode()
            [iteration_idx_3D,iteration_idx2D] = findWorldPointsInView(worldPointSet,...
                linked_IDs(i));
        else
            [interation_generated_idx_3D,interation_generated_idx_2D] = ...
                findWorldPointsInView(worldPointSet,linked_IDs(i));
            iteration_idx_3D = interation_generated_idx_3D{1};
            iteration_idx2D = interation_generated_idx_2D{1};
        end
        Points3D = worldPointSet.WorldPoints(iteration_idx_3D,:);
        
        differences = Points3D - iteration_pose.Translation;  % Compute the vector differences
        squaredDistances = sum(differences.^2, 2);     % Sum the squares of components
        euclideanDistances = sqrt(squaredDistances);   % Square root of summed squares
        med_depth = median(euclideanDistances);      % Median of the distances

        if norm(iteration_pose.Translation - currPose.Translation)/med_depth < 0.01
            continue
        end

        iteration_features = keyframeSet.Views.Features{linked_IDs(i)};
        iteration_points = keyframeSet.Views.Points{linked_IDs(i)};
        iteration_locations = iteration_points.Location;
        iteration_scales = iteration_points.Scale;

        if isSimMode()
            [~,currIdx2D] = findWorldPointsInView(worldPointSet,currKeyframeID);
        else
            [~,iterIdx2D_generated] = findWorldPointsInView(worldPointSet,currKeyframeID);
            currIdx2D = iterIdx2D_generated{1};
        end

        idxs_unmatched_prev = setdiff(uint32(1:size(iteration_features,1))',...
            iteration_idx2D, 'stable');
        idxs_unmatched_curr = setdiff(uint32(1:size(currFeatures,1))',...
            currIdx2D, 'stable');

        features_unmatched_prev = iteration_features(idxs_unmatched_prev,:);
        features_unmatched_curr = currFeatures(idxs_unmatched_curr,:);

        locations_unmatcced_prev = iteration_locations(idxs_unmatched_prev,:);
        locations_unmatcced_curr = currLocations(idxs_unmatched_curr,:);

        scales_unmatched_prev = iteration_scales(idxs_unmatched_prev);
        scales_unmatched_curr = currScales(idxs_unmatched_curr);

        idxPairs = matchFeatures(binaryFeatures(features_unmatched_prev),...
            binaryFeatures(features_unmatched_curr),'Unique',true,'MaxRatio',...
            0.7, 'MatchThreshold',40);

        if isempty(idxPairs)
            continue
        end

        matchedPoints_prev = locations_unmatcced_prev(idxPairs(:,1),:);
        matchedPoints_curr = locations_unmatcced_curr(idxPairs(:,2),:);

        % epipole in current keyframe
        epiPole = world2img(iteration_pose.Translation,pose2extr(currPose),intrinsics);
        distToEpipole = vecnorm(matchedPoints_curr - epiPole, 2, 2);

        % compute F
        F = computeF(intrinsics,iteration_pose,currPose);

        % epipolar line in second image
        epiLine = epipolarLine(F,matchedPoints_curr);
        distToLine = abs(sum(epiLine.* [matchedPoints_prev ones(size(matchedPoints_prev,1),1)], 2))./sqrt(sum(epiLine(:,1:2).^2,2));
        isCorrect = distToLine < 2*scales_unmatched_curr(idxPairs(:,2)) & ...
            distToEpipole > 10*scales_unmatched_curr(idxPairs(:,2));

        idxPairs = idxPairs(isCorrect,:);
        matchedPoints_prev = matchedPoints_prev(isCorrect,:);
        matchedPoints_curr = matchedPoints_curr(isCorrect,:);

        % check for sifficient parallax
        isposeDifferent = isSufficientParallax(matchedPoints_prev,matchedPoints_curr,iteration_pose,...
            currPose,intrinsics,min_Parallax);
        
        matchedPoints_prev = matchedPoints_prev(isposeDifferent,:);
        matchedPoints_curr = matchedPoints_curr(isposeDifferent,:);
        idxPairs = idxPairs(isposeDifferent,:);

        iteration_proj_mat = cameraProjection(intrinsics,pose2extr(iteration_pose));
        
        % triangulate two views to find world points
        [Points3D, error_reproj, validIdx] = triangulate(matchedPoints_prev,...
            matchedPoints_curr,iteration_proj_mat,currProjMatrix);

        inlier = filterTriangulatedMapPoints(Points3D, iteration_pose, currPose, ...
        scales_unmatched_prev(idxPairs(:,1)), scales_unmatched_curr(idxPairs(:,2)), ...
        error_reproj, scaleFactor, validIdx);

        if any(inlier)
            Points3D = Points3D(inlier,:);
            idxPairs = idxPairs(inlier,:);

            map_indices_prev = idxs_unmatched_prev(idxPairs(:,1));
            map_indices_curr = idxs_unmatched_curr(idxPairs(:,2));

            [worldPointSet, idxs] = addWorldPoints(worldPointSet,Points3D);
            newPointIdx       = [newPointIdx; idxs];

            worldPointSet = addCorrespondences(worldPointSet,linked_IDs(i),idxs,map_indices_prev);
            worldPointSet = addCorrespondences(worldPointSet,currKeyframeID,idxs,map_indices_curr);

            if isSimMode()
                [~,index_past] = intersect(keyframeSet.Connections{:,1:2},...
                    [linked_IDs(i),currKeyframeID],'row','stable');
                matches_before = keyframeSet.Connections.Matches{index_past};
            else
                connections = keyframeSet.Connections;
                [~,index_past] = intersect([connections.ViewId1, connections.ViewId2],...
                    [linked_IDs(i),currKeyframeID],'row','stable');
                matches_before = connections.Matches{index_past};
            end
            matchesUpdated = [matches_before;map_indices_prev,map_indices_curr];
            keyframeSet = updateConnection(keyframeSet,linked_IDs(i),currKeyframeID,...
                'Matches',matchesUpdated);
        end
    end
end


function validPoints = filterTriangulatedMapPoints(worldCoords, frame1Pose, frame2Pose, ...
    scaleFrame1, scaleFrame2, projectionErrors, scaleMultiplier, pointsVisibility)

% Calculate vector from camera to points for each frame
vectorToPoints1 = worldCoords - frame1Pose.Translation;
vectorToPoints2 = worldCoords - frame2Pose.Translation;

% Compute Euclidean distances from camera to points
distToPoints1 = vecnorm(vectorToPoints1, 2, 2);
distToPoints2 = vecnorm(vectorToPoints2, 2, 2);

% Calculate distance and scale ratios
distanceRatio = distToPoints1 ./ distToPoints2;
scaleRatio = scaleFrame2 ./ scaleFrame1;

% Define consistency threshold based on scale factor
consistencyThreshold = 1.5 * scaleMultiplier;

% Check scale and distance consistency
consistentScale = (distanceRatio ./ scaleRatio < consistencyThreshold | ...
    scaleRatio ./ distanceRatio < consistencyThreshold);

% Define maximum allowable reprojection error
allowedError = sqrt(6);
isErrorSmall = projectionErrors < allowedError * min(scaleFrame1, scaleFrame2);

% Determine valid points based on scale consistency, error size, and visibility
validPoints = consistentScale & isErrorSmall & pointsVisibility;
end

function significantParallax = isSufficientParallax(firstPoints, secondPoints, firstPose, secondPose, cameraParams, minParallaxAngle)

% Parallax check: Compute rays originating from each camera to the points in homogeneous coordinates
rayFromFirstCamera = [firstPoints, ones(size(firstPoints(:, 1)))] / cameraParams.K' * firstPose.R';
rayFromSecondCamera = [secondPoints, ones(size(secondPoints(:, 2)))] / cameraParams.K' * secondPose.R';

% Calculate the cosine of the parallax angle between the two rays
cosineOfParallax = sum(rayFromFirstCamera .* rayFromSecondCamera, 2) ./ ...
    (vecnorm(rayFromFirstCamera, 2, 2) .* vecnorm(rayFromSecondCamera, 2, 2));

% Determine if the parallax is significant
significantParallax = cosineOfParallax < cosd(minParallaxAngle) & cosineOfParallax > 0;
end

function tf = isSimMode()
    tf = isempty(coder.target);
end



