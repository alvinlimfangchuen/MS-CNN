clear; clc;
inputSize = [224 224 3];
imdsTrain = imageDatastore("D:\Database\iHGS_Database\IMAGE_MHI_COMBINE_2_LATEST","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain,0.8);
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,'ColorPreprocessing','gray2rgb');
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation, 'ColorPreprocessing','gray2rgb');


layers = [
    imageInputLayer([224 224 3],"Name","input")
    convolution2dLayer([3 3],64,"Name","conv1_1","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu1_1")
    convolution2dLayer([3 3],64,"Name","conv1_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu1_2")
    maxPooling2dLayer([2 2],"Name","pool1","Stride",[2 2])
    convolution2dLayer([3 3],128,"Name","conv2_1","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu2_1")
    convolution2dLayer([3 3],128,"Name","conv2_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu2_2")
    maxPooling2dLayer([2 2],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],256,"Name","conv3_1","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu3_1")
    convolution2dLayer([3 3],256,"Name","conv3_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu3_2")
    convolution2dLayer([3 3],256,"Name","conv3_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu3_3")
    maxPooling2dLayer([2 2],"Name","pool3","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv4_1","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu4_1")
    convolution2dLayer([3 3],512,"Name","conv4_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu4_2")
    convolution2dLayer([3 3],512,"Name","conv4_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu4_3")
    maxPooling2dLayer([2 2],"Name","pool4","Stride",[2 2])
    convolution2dLayer([3 3],512,"Name","conv5_1","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu5_1")
    convolution2dLayer([3 3],512,"Name","conv5_2","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu5_2")
    convolution2dLayer([3 3],512,"Name","conv5_3","Padding",[1 1 1 1],"WeightL2Factor",0)
    reluLayer("Name","relu5_3")
    maxPooling2dLayer([2 2],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","WeightL2Factor",0)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","WeightL2Factor",0)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(100,"Name","fc8","WeightL2Factor",0)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
    
%Training Options
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',0.0001, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,layers,options);

%----Get Confusion Matrix----%
% Predict on validation set
predictedLabels = classify(net, augimdsValidation);
trueLabels = imdsValidation.Labels;
% Compute confusion matrix
confusion = confusionmat(trueLabels, predictedLabels);
% Display confusion matrix
cm = confusionchart(trueLabels, predictedLabels);

%----Compute evaluation metrics----%
num_classes = size(confusion, 1);
num_samples = sum(sum(confusion));
precision = zeros(num_classes, 1);
specificity = zeros(num_classes, 1);
sensitivity = zeros(num_classes, 1);

accuracy = sum(diag(confusion)) / num_samples;
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    % calculate true positives, false positives, false negatives, and true negatives
    tp = confusion(i, i);
    fp = sum(confusion(:, i)) - tp;
    fn = sum(confusion(i, :)) - tp;
    tn = num_samples - tp - fp - fn;
    
    % calculate precision, specificity, sensitivity/recall, and F1-score
       
    precision(i) = tp / (tp + fp);
    specificity(i) = tn / (tn + fp);
    sensitivity(i) = tp / (tp + fn);
    f1_score(i) = 2 * precision(i) * sensitivity(i) / (precision(i) + sensitivity(i));
    
    
end

% Compute mean values, ignoring NaNs
mean_precision = nanmean(precision);
mean_specificity = nanmean(specificity);
mean_sensitivity = nanmean(sensitivity);
mean_f1_score = nanmean(f1_score);

%Display
output = sprintf('Accuracy:%.4f \nPrecision:%.4f \nSpecificity:%.4f \nSensivity:%.4f \nF1-Score:%.4f',accuracy, mean_precision, mean_specificity, mean_sensitivity,mean_f1_score);
disp(output);

out = sprintf('\nTP:%d TN:%d FP:%d FN:%d', tp, tn, fp, fn);
disp(out);