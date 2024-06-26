function options=training(imdsValid1)
options=trainingOptions...
    'SquaredGradientDecayFactor',0.95, ...
    'MiniBatchSize',16, ...
    'MaxEpochs',, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValid1, ...
    'InitialLearnRate' ,0.0001, ... 
    'L2Regularization',0.001, ...
    'GradientThresholdMethod','l2norm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.4, ...
   'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');
end