function [feature_descriptors,valid_points] = generateFeatures(img,scaleFactor,numLevels,numFeatures)
    %INPUTS
        %img: image used for feature detection
        %scale factor: the factor of size reduction of the image between
        %pyramid levels during feature detection
        %numLevels:number of pyramid levels used in feature detection
        %numFeatures: number of features to output
    %OUTPUTS
        %feature_descriptors: feature descpitors corresponding to valid
        %points (only points for which extractFeatures could find valid
        %points
        %valid_points: list of image coordinates for the features
        %extracted.  These points correspond to the feature descriptors
        %that will be used for matching
    img_gray = im2gray(img);

    features_all = detectORBFeatures(img_gray,ScaleFactor=scaleFactor,NumLevels=numLevels);

    img_size = size(img_gray,[1,2]);

    features_uniform = selectUniform(features_all,numFeatures, img_size);
    
    [feature_descriptors, valid_points] = extractFeatures(img_gray, features_uniform);

end