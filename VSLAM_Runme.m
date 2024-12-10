%% VSLAM Runme
% Whit Whittall, Nicholas Martinez
% wrapper script for VSLAM pipeline

%% Save and Unpack Dataset

% download data from TUM RGB-D Benchmark
dataURL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"; 
dataFolder = fullfile(tempdir, 'tum_rgbd_dataset', filesep);
tgzFileName = [dataFolder, 'fr3_office.tgz'];

% Create a folder in a temporary directory to save the downloaded file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this may be unnecessary
if ~exist(dataFolder, "dir")
    mkdir(dataFolder); 
    disp('Downloading fr3_office.tgz. Be patient, this can take a minute.') 
    websave(tgzFileName, dataURL);

    % Extract contents of the downloaded file
    disp('Extracting fr3_office.tgz ...') 
    untar(tgzFileName, dataFolder); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create imageDatastore object to store the images
imageFolder   = [dataFolder,'rgbd_dataset_freiburg3_long_office_household/rgb/'];
imds          = imageDatastore(imageFolder);

% show first image to confirm function of data retrieval
currFrame = 1;
currImg = readimage(imds, currFrame);
himage = imshow(currImg);

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
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 1000;

[preFeatures, prePoints] = extractORBFeatures(currImg, numPoints);

currFrame = currFrame + 1;
firstImg = currImg; % Preserve the first frame 

% enter map initialization loop
isMapInitialized = false;

while ~isMapInitialized && currFrame < length(imds.Files)
    % get features for next frame
    currImg = readimage(imds, currFrame);
    [currFeatures, currPoints] = extractORBFeatures(currImg, numPoints);
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
    
end