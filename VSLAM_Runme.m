%% VSLAM Runme
% Whit Whittall, Nicholas Martinez
% wrapper script for VSLAM pipeline

%% Save and Unpack Dataset

% download data from TUM RGB-D Benchmark
dataURL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"; 
dataFolder = fullfile(tempdir, 'tum_rgbd_dataset', filesep);
tgzFileName = [dataFolder, 'fr3_office.tgz'];

% Create a folder in a temporary directory to save the downloaded file
if ~exist(dataFolder, "dir")
    mkdir(dataFolder); 
    disp('Downloading fr3_office.tgz. Be patient, this can take a minute.') 
    websave(tgzFileName, dataURL);

    % Extract contents of the downloaded file
    disp('Extracting fr3_office.tgz ...') 
    untar(tgzFileName, dataFolder); 
end

% Create imageDatastore object to store the images
imageFolder = [dataFolder,'rgbd_dataset_freiburg3_long_office_household/rgb/'];
imds = imageDatastore(imageFolder);

% show first image to confirm function of data retrieval
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% might be nicer to have script store these images
% for now I've included breakpoints at each checkpoint
currFrame = 1;
currImg = readimage(imds, currFrame);
himage = imshow(currImg);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize Map
% Set random seed for reproducibility
rng(0);

% store camera intrinsics in cameraIntrinsics object
% intrinsics for dataset at:
% https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% images in this dataset are already undistorted which slightly streamlines
% implementation
focalLength = [535.4, 539.2];       % in units of pixels
principalPoint = [320.1, 247.6];    % in units of pixels
imageSize = size(currImg,[1 2]);      % in units of pixels
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);

% detect and extract orb features from first frame in set
% scaleFactor = 1.2;
% numLevels   = 8;
numPoints   = 1000;

[preFeatures, prePoints] = extractORBFeatures(currImg, numPoints);

currFrame = currFrame + 1;
firstImg = currImg; % Preserve the first frame 

% enter map initialization loop
isMapInitialized = false;

while ~isMapInitialized && currFrame < length(imds.Files)
    % get features from current frame
    currImg = readimage(imds, currFrame);
    [currFeatures, currPoints] = extractORBFeatures(currImg, numPoints);

    % incriment frame index
    currFrame = currFrame + 1;

    % find likely feature matches
    matches = matchFeatures(preFeatures, currFeatures, Unique=true, MaxRatio=0.9, MatchThreshold=40);

    % if not enough matches found, check the next frame
    if length(matches) < 100    % curious how varying this threshold affects performance
        continue
    end

    % compute fundamental matrix
    % some implimentations compute both homography and fundamental matrix,
    % and compare the two based on a heuristic to select which transform to
    % use
    % this can improve performance in spaces with many planar surfaces
    preMatches = prePoints(matches(:,1),:);
    currMatches = currPoints(matches(:,2),:);
    % because we know camera intrinsics, we use estimateEssentialMatrix() 
    % rather than estimateFundamentalMatrix
    [F, inliersF] = getFundamentalMatrix(preMatches, currMatches, intrinsics);

    % compute relative camera pose
    inlierPrePoints = preMatches(inliersF);
    inlierCurrPoints = currMatches(inliersF);
    [relPose, valFrac] = estrelpose(F, intrinsics, inlierPrePoints, inlierCurrPoints);

    % if less than 90% of inlier points project in front of both cameras, F
    % is likely incorrect
    if valFrac < 0.9 || numel(relPose) > 1
        continue
    end
    
    % triangulate 3D world points from two views
    % assume first view is at origin, second view (relPose) is relative to
    % origin
    [worldPoints, worldInliers, isValid] = triangulateWorldPoints(rigidtform3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics);

    % if there is not sufficient parallax between views, check next frame
    if ~isValid
        continue
    end

    % get original index of features in the two keyframes
    matches = matches(inliersF(worldInliers), :);

    isMapInitialized = true;

    disp(['Map initialized with frame 1 and frame ', num2str(currFrame - 1)])
end

% close previous figure and show matched features
if isMapInitialized
    close(himage.Parent.Parent);
    hfeature = showMatchedFeatures(firstImg, currImg, prePoints(matches(:, 1)), currPoints(matches(:, 2)), "montage");
else
    error('map initialization failed')
end

%% Store initial keyframes and world points
% CV toolbox uses imageviewset and worldpointset objects to store store
% keyframes and their associated data and world points and their 2D
% correspondences

% initialize empty objects
keyframeSet = imageviewset;
worldPointSet = worldpointset;

% add first keyframe; place camera at origin
preViewID = 1;
keyframeSet = addView(keyframeSet, preViewID, rigidtform3d, Points = prePoints, Features = preFeatures.Features);

% add second keyframe
currViewID = 2;
keyframeSet = addView(keyframeSet, currViewID, relPose, Points = currPoints, Features = currFeatures.Features);

% add connection between the first and the second keyframes
keyframeSet = addConnection(keyframeSet, preViewID, currViewID, relPose, Matches = matches);

% add 3D world points
[worldPointSet, newPointIdx] = addWorldPoints(worldPointSet, worldPoints);

% add 3D-2D point correspondences from first keyframe
worldPointSet = addCorrespondences(worldPointSet, preViewID, newPointIdx, matches(:, 1));

% add 3D-2D point correspondences from second keyframe
worldPointSet = addCorrespondences(worldPointSet, currViewID, newPointIdx, matches(:, 2));

%% Initialize place recognition database for loop closure detection
% many VSLAM techniques, ORBSLAM included, utilize the Bag of Words (BoW)
% algorithm for loop closure detection

% bagOfFeaturesDBoW object CV toolbox uses takes a long time to create
if ~exist("bag", "var")
    disp('Creating Bag of Words. Please be patient, this takes a long time.')
    bag = bagOfFeaturesDBoW(imds);
end

% initialize loop closure detection database
loopDatabase = dbowLoopDetector(bag);

% Add features of the first two key frames to the database
addVisualFeatures(loopDatabase, preViewID, preFeatures);
addVisualFeatures(loopDatabase, currViewID, currFeatures);

%% Refine and visualize initial reconstruction

% perform bundle adjustment on first two keyframes
% note that PCG solver is from g2o library, which ORBSLAM also uses
tracks = findTracks(keyframeSet);
camPoses = poses(keyframeSet);
% [refinedPoints, refinedPoses] = bundleAdjustment(worldPoints, tracks, ...
%     camPoses, intrinsics, FixedViewIDs = 1, PointsUndistorted = true, ...
%     AbsoluteTolerance = 1e-7, RelativeTolerance = 1e-15, ...
%     MaxIterations = 20, Solver = "preconditioned-conjugate-gradient");
[refinedPoints, refinedPoses] = bundleAdjustment(worldPoints, tracks, ...
    camPoses, intrinsics, FixedViewIDs = 1, PointsUndistorted = true, ...
    AbsoluteTolerance = 1e-7, RelativeTolerance = 1e-15, ...
    MaxIterations = 20, Solver = "preconditioned-conjugate-gradient");

% scale map and camera pose using median world point depth
medDepth   = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medDepth;

refinedPoses.AbsolutePose(currViewID).Translation = refinedPoses.AbsolutePose(currViewID).Translation / medDepth;
relPose.Translation = relPose.Translation/medDepth;

% update keyframes with refined poses
keyframeSet = updateView(keyframeSet, refinedPoses);
keyframeSet = updateConnection(keyframeSet, preViewID, currViewID, relPose);

% update map with refined world points
worldPointSet = updateWorldPoints(worldPointSet, newPointIdx, refinedPoints);

% update view directions and depth
worldPointSet = updateLimitsAndDirection(worldPointSet, newPointIdx, keyframeSet.Views);

% update representative view
worldPointSet = updateRepresentativeView(worldPointSet, newPointIdx, keyframeSet.Views);

% Visualize matched features in the current frame
% will use class copied from MathWorks CV toolbox example for visualization
close(hfeature.Parent.Parent);
featurePlot = helperVisualizeMatchedFeatures(currImg, currPoints(matches(:,2)));

% Visualize initial map points and camera trajectory
mapPlot = helperVisualizeMotionAndStructure(keyframeSet, worldPointSet);

% Show legend
showLegend(mapPlot);

%% Main Loop

% initialize viewIDs for main loop
currKeyframeID = currViewID;
lastKeyframeID = currViewID;

% initialize frame indexes for main loop
lastKeyframe = currFrame - 1;
addedFrames = [1, lastKeyframe];

loopClosed = false;
isLastKeyframe = true;

% create and initialize Kanade-Lucas_Tomasi feature tracker
% KLT tracker used to predict match locations from last frame in current
% frame assuming linear motion model for camera
tracker = vision.PointTracker(MaxBidirectionalError = 5);
initialize(tracker, currPoints.Location(matches(:,2), :), currImg);

while currFrame < length(imds.Files)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Main Loop:
    % for each frame
        % 1) get ORB features and match with features in last keyframe with
        % corresponding 3D world points. This gets us 3D-2D point
        % correspondences in current frame.
        % 2) use 3D-2D point correspondences to estimate camera pose with
        % PnP and refine with motion-only bundle adjustment
        % 3) search for more 3D-2D correspondences by projecting nearby
        % world points into camera frame, refine camera pose again with
        % bundle adjustment
        % 4) decide if current frame is keyframe. If yes, continue to local
        % mapping and loop closure detection. Otherwise, continue to next
        % frame.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % get ORB features for current frame
    currImg = readimage(imds, currFrame);
    [currFeatures, currPoints] = extractORBFeatures(currImg, numPoints);

    % track last keyframe
    [currPose, worldPointsIdx, featureIdx] = trackLastKeyframe(...
        tracker, currImg, worldPointSet, keyframeSet.Views, currFeatures, ...
        currPoints, lastKeyframeID, intrinsics);

    % check if current frame is keyframe
    [localKeyframeIDs, currPose, worldPointsIdx, featureIdx, isKeyframe] = ...
        stockKeyframeSelector(worldPointSet, keyframeSet, worldPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, ...
        isLastKeyframe, lastKeyframe, currFrame);

end