clear,close    
clear all;
close all;
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Loading Dataset 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datapath='C:\Users\mumtaz\Research Work\New Paper\Combined DataSet\';
% Image     
imds=imageDatastore(datapath, ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
% Determine the split up
total_split=countEachLabel(imds  )
num_images=length(imds.Labels);
% Visualize random 20 images
perm=randperm(num_images,2);
train_percent=0.80;
[imdsTrain_,imdsTest]=splitEachLabel(imds,train_percent,'randomize');
% Split the Training and Validation
valid_percent=0.1;
[imdsValid,imdsTrain]=splitEachLabel(imdsTrain_,valid_percent,'randomize');
train_split=countEachLabel(imdsTrain);

 numClasses=numel(categories(imdsTrain.Labels));
 
 net1 = custom_ResNet_Modified_VGG1();
 net2 = custom_ResNet_Modified_VGG2();

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Preprocessing 
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Imp: CE_MRI already normalized while saving from the source site

% Objective No. 1: Reducing channels from 3 to 1
%               2: Resiziing 512 to 224
% % % 
imdsTrain_.ReadFcn=@(filename)preprocess_images(filename,[net1.Layers(1).InputSize(1), net1.Layers(1).InputSize(2)]);
imdsTrain.ReadFcn=@(filename)preprocess_images(filename,[net1.Layers(1).InputSize(1), net1.Layers(1).InputSize(2)]);
imdsValid.ReadFcn=@(filename)preprocess_images(filename,[net1.Layers(1).InputSize(1), net1.Layers(1).InputSize(2)]);
imdsTest.ReadFcn=@(filename)preprocess_images(filename,[net1.Layers(1).InputSize(1), net1.Layers(1).InputSize(2)]);
% % %%%%%%%%%%%%%%%%%%%%%
% % %% Data Augmentation
% % %%%%%%%%%%%%%%%%%%%%%
augmenter = imageDataAugmenter( ...
'RandRotation',[-5 5],'RandXReflection',1,...
'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
% imdsTrain1 = augmentedImageDatastore([224,224],imdsTrain,'DataAugmentation',augmenter);


% %     % Resizing all training images to [224 224] for ResNet architecture
    imdsTrain_1 = augmentedImageDatastore([227,227 1],imdsTrain_);
    imdsTrain1 = augmentedImageDatastore([227,227 1],imdsTrain,'DataAugmentation',augmenter);
    imdsValid1= augmentedImageDatastore([227,227 1],imdsValid,'DataAugmentation',augmenter);
    imdsTest1 = augmentedImageDatastore([227,227 1],imdsTest);

%%%%%%%%%%%%%%%%%%%%%
%%  Loading FT-CNN with options
%%%%%%%%%%%%%%%%%%%%%
options=training1(imdsValid1);%changed from imdsValid1

trainingLabels = imdsTrain_.Labels;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  FT-CNN training (weight assignment)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Proposed_RESNETVGG1=trainNetwork(imdsTrain1,net1,options)%changed from imdsTrain1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% TFT-CNN testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[predicted_labels_Net1,posterior1]=classify(Proposed_RESNETVGG1,imdsTest);%changed from imdsTest1

Predicted_labels_network1=predicted_labels_Net1;
Test_labels_network=imdsTest.Labels;
 Network_accuracy_1 = sum(predicted_labels_Net1 == Test_labels_network)/numel(Test_labels_network)
% % % % % Confusion Matrix
 figure;
 plotconfusion(Test_labels_network,predicted_labels_Net1)
