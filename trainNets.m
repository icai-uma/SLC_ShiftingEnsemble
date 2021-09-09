% Code for training the networks

networks = {'googlenet', 'mobilenetv2'};

% Load previously created ImageDatastore
load('HAM10000.mat');

classes = numel(categories(ds.Labels));

aug = imageDataAugmenter('RandXReflection',true, 'RandYReflection',true, 'RandRotation',@() randi(4)*90);

for n = 1:length(networks)
    
    net = layerGraph(eval(networks{n}));
    net = replaceLayer(net, net.Layers(end-2).Name, fullyConnectedLayer(classes, 'Name','my_predictions', 'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20));
    net = replaceLayer(net, net.Layers(end).Name, classificationLayer('Name','my_classification'));

    for i=1:length(trainSet)
        trainDS = augmentedImageDatastore(net.Layers(1).InputSize, trainSet{i}, 'DataAugmentation', aug);
        valDS = augmentedImageDatastore(net.Layers(1).InputSize, valSet{i}, 'DataAugmentation', aug);

         options = trainingOptions('sgdm',...
            'MiniBatchSize',16,...
            'MaxEpochs',10,...
            'InitialLearnRate',1e-4,...
            'Shuffle','every-epoch',...
            'ExecutionEnvironment', 'multi-gpu',...
            'DispatchInBackground', true,...
            'ValidationData',valDS,...
            'ValidationFrequency',10);

        % Training
        tic
        [net_train, info] = trainNetwork(trainDS, net, options);
        time = toc;

        % Accuracy calculation
        tic
        y = net_train.classify(augmentedImageDatastore(net.Layers(1).InputSize, trainSet{i}));
        timeSimTrain = toc;
        accTrain = sum(y == trainSet{i}.Labels)/numel(trainSet{i}.Labels);
        tic
        y = net_train.classify(augmentedImageDatastore(net.Layers(1).InputSize, valSet{i}));
        timeSimVal = toc;
        accVal = sum(y == valSet{i}.Labels)/numel(valSet{i}.Labels);
        tic
        y = net_train.classify(augmentedImageDatastore(net.Layers(1).InputSize, testSet{i}));
        timeSimTest = toc;
        accTest = sum(y == testSet{i}.Labels)/numel(testSet{i}.Labels);
        tic
        y = net_train.classify(augmentedImageDatastore(net.Layers(1).InputSize, ds));
        timeSim = toc;
        accuracy = sum(y == ds.Labels)/numel(ds.Labels);

        % Confusion matrix calculation
        confusion = confusionmat(ds.Labels, y);
 
        save(sprintf('p%i_%s_%i', classes, networks{n}, i))
    end
end

