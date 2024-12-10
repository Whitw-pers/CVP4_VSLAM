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

% detect and extract orb features
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 1000;

[preFeatures, prePoints] = extractORBFeatures(currImg, scaleFactor, numLevels, numPoints);

currFrame = currFrame + 1;
firstImg       = currImg; % Preserve the first frame 

% enter map initialization loop
isMapInitialized  = false;

while ~isMapInitialized && currFrame < length(imds.Files)
end