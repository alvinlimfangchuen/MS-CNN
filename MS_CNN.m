clear; clc;
%Load Dataset
imdsTrain = imageDatastore("D:\Database\iHGS_Database\IMAGE_MHI_COMBINE_2_LATEST","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);
augimdsTrain = augmentedImageDatastore([224 224 1],imdsTrain);
augimdsValidation = augmentedImageDatastore([224 224 1],imdsValidation);


%% Design
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([224 224 1],"Name","input")
    convolution2dLayer([3 3],32,"Name","conv_1")
    batchNormalizationLayer("Name","bn_1")
    reluLayer("Name","relu_1") %Activation Function
    maxPooling2dLayer([2 2],"Name","maxpool_1_2","Stride",[2 2])
    convolution2dLayer([3 3],64,"Name","conv_2_1")
    batchNormalizationLayer("Name","bn_2_1")
    reluLayer("Name","relu_2_1")%Activation Function
    maxPooling2dLayer([2 2],"Name","maxpool_1_1","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_2_3_1","Padding","same")
    batchNormalizationLayer("Name","bn_2_3_1")
    reluLayer("Name","relu_2_3_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_2_2_1","Padding","same")
    batchNormalizationLayer("Name","bn_2_2_1")
    reluLayer("Name","relu_2_2_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_1")
    maxPooling2dLayer([2 2],"Name","maxpool_1_3_1","Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","conv_2_3_2","Padding","same")
    batchNormalizationLayer("Name","bn_2_3_2")
    reluLayer("Name","relu_2_3_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([5 5],64,"Name","conv_2_2_2","Padding","same")
    batchNormalizationLayer("Name","bn_2_2_2")
    reluLayer("Name","relu_2_2_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat_2")
    maxPooling2dLayer([2 2],"Name","maxpool_1_3_2","Stride",[2 2])
    fullyConnectedLayer(1024,"Name","fc_1")
    dropoutLayer(0.5,"Name","drop6")
    reluLayer("Name","relu7")
    fullyConnectedLayer(100,"Name","fc_2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;


lgraph = connectLayers(lgraph,"maxpool_1_1","conv_2_3_1");
lgraph = connectLayers(lgraph,"maxpool_1_1","conv_2_2_1");
lgraph = connectLayers(lgraph,"relu_2_3_1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"relu_2_2_1","depthcat_1/in2");
lgraph = connectLayers(lgraph,"maxpool_1_3_1","conv_2_3_2");
lgraph = connectLayers(lgraph,"maxpool_1_3_1","conv_2_2_2");
lgraph = connectLayers(lgraph,"relu_2_2_2","depthcat_2/in1");
mscnn = connectLayers(lgraph,"relu_2_3_2","depthcat_2/in2");

%%

%Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',0.001, ...
    'Shuffle','every-epoch', ...
    'ExecutionEnvironment','gpu', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',false, ...
    'Plots','training-progress');

%Remove this comment to train the model
%net = trainNetwork(augimdsTrain,lgraph,options);

% Enter the following code at matlab command window to view the architecture details
% analyzeNetwork(mscnn);

