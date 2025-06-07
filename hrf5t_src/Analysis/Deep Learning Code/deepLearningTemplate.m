% A template script for the deep learning procedure used in "Categorization
% and Recognition in a Naturalistic Stimulus Domain." File directories are
% left blank, to be filled in by the user.
%
% The script creates an output folder containing various iterations of the 
%
% Date: 8/14/22
% Author: Brian J. Meagher
%% Preliminaries
rng('shuffle');
clear, clc

maxTime = 3*24*60*60;
maxIter = 100;

% Output folder for current run. Creates a folder in current working
% directory to save results in, named based on the current timestamp.
outputFolderName = char(datetime('now', 'Format', 'yyyyMMddHHmmss'));
mkdir(outputFolderName);

%% Loading data
% Standard image data
image_location_standard = []; % Insert path to standard stimuli here
files = dir([image_location_standard '*.png']);
numRocksStandard = numel(files);
namesStandard = cell(numRocksStandard, 1);
for i = 1:numRocksStandard
    namesStandard{i} = [image_location_standard files(i).name];
end

% Photoshopped image data
image_location_photoshopped = []; % Insert path to HSN stimuli here
files = dir([image_location_photoshopped '*.png']);
numRocksPhotoshopped = numel(files);
namesPhotoshopped = cell(numRocksPhotoshopped, 1);
for i = 1:numRocksPhotoshopped
    namesPhotoshopped{i} = [image_location_photoshopped files(i).name];
end

% Combine some things
numRocks = numRocksStandard + numRocksPhotoshopped;
names = [namesStandard; namesPhotoshopped];

% Categories/subtypes
categories = categorical([repmat("Igneous", 160, 1)
              repmat("Metamorphic", 160, 1)
              repmat("Sedimentary", 160, 1)
              repmat("Igneous", 20, 1)
              repmat("Metamorphic", 20, 1)
              repmat("Sedimentary", 20, 1)]);
rockNames = ["Andesite", "Basalt", "Diorite", "Gabbro", "Granite", ...
             "Obsidian", "Pegmatite", "Peridotite", "Pumice", "Rhyolite", ...
            "Amphibolite", "Anthracite", "Gneiss", "Hornfels", "Marble", ...
            "Migmatite", "Phyllite", "Quartzite", "Schist", "Slate", ...
            "Bituminous Coal", "Breccia", "Chert", "Conglomerate", ...
            "Dolomite", "Micrite", "Rock Gypsum", "Rock Salt", ...
            "Sandstone", "Shale"];
subtypes = categorical([repelem(rockNames, 16) repelem(rockNames, 2)]');

% Make a table
data = table(names, categories, subtypes);

% Limit to standard stimuli
data = data(1:numRocksStandard,:);
subtypes = subtypes(1:numRocksStandard);

%% Preprocessing and group assignment

% Define proportion of images assigned to each group
trainProp = 5/6;
valProp = 1/6;

% Using cvpartition function to stratify the sample
c = cvpartition(subtypes, 'HoldOut', valProp);
trainIx = training(c);
valIx = test(c);

trainTbl = data(trainIx,:);
valTbl = data(valIx,:);

% Augmentation options
augs = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [0, 360], ...
    'RandScale', [0.5, 2] ...
    );

% Augmented datastores
trainDs = augmentedImageDatastore([224 224],trainTbl,'categories','DataAugmentation',augs);
valDs = augmentedImageDatastore([224 224],valTbl,'categories','DataAugmentation',augs);

%% Set up parameters for Bayesian optimization
optimVars = [
    optimizableVariable('numberFClayers', [1 5], 'Type', 'integer')
    optimizableVariable('numberFCunits', [1 3000], 'Type', 'integer')
    optimizableVariable('minibatchSize', [1 100], 'Type', 'integer')
    optimizableVariable('InitialLearnRate',[1e-4, 1], 'Transform', 'log')];

ObjFcn = makeObjFcn(trainDs, valDs, valTbl, outputFolderName);
BayesObject = bayesopt(ObjFcn, optimVars, ...
    'MaxTime', maxTime, ...
    'MaxObjectiveEvaluations', maxIter, ...
    'IsObjectiveDeterministic', false, ...
    'UseParallel', false, ...
    'SaveFileName', [outputFolderName '/BayesoptResults.mat'], ...
    'OutputFcn', {@saveToFile});

%% Set objective function
function ObjFcn = makeObjFcn(trainDs, valDs, valTbl, outputFolderName)
    ObjFcn = @valErrorFun;
    function[valError,cons,fileName] = valErrorFun(optVars)
        % Set up the network
        net = resnet50;
        lgraph = layerGraph(net);
        lgraph = removeLayers(lgraph,{'fc1000_softmax','ClassificationLayer_fc1000'});
        layers = lgraph.Layers;
        connections = lgraph.Connections;
        layers = freezeWeights(layers);
        lgraph = createLgraphUsingConnections(layers, connections);
        
        numCats = 3;
        newLayers = [createFClayers(optVars.numberFClayers, numCats, optVars.numberFCunits)
                     classificationLayer('Name', 'output')];
        lgraph = replaceLayer(lgraph,'fc1000',newLayers);
        
        % Set options and train networok
        % Step one: All but added layers frozen
        options1 = trainingOptions('adam','InitialLearnRate',optVars.InitialLearnRate, ...
                           'Plots', 'training-progress', ...
                           'MaxEpochs', 500, ...
                           'ValidationData', valDs, ...
                           'ValidationFrequency', floor(trainDs.NumObservations/optVars.minibatchSize), ...
                           'ValidationPatience', 20, ...
                           'MiniBatchSize', optVars.minibatchSize, ...
                           'ExecutionEnvironment', 'multi-gpu', ...
                           'Verbose',false);
        
        rockNet = trainNetwork(trainDs,lgraph,options1);
        close(findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE'))

        % Step two: All layers unfrozen
        lgraph = layerGraph(rockNet);
        layers = lgraph.Layers;
        connections = lgraph.Connections;
        layers = unfreezeWeights(layers);
        lgraph = createLgraphUsingConnections(layers, connections);
        
        % Set options and train network
        options2 = trainingOptions('sgdm','InitialLearnRate', 1e-4, ...
                           'Momentum', 0.9, ...
                           'Plots', 'training-progress', ...
                           'MaxEpochs', 500, ...
                           'ValidationData', valDs, ...
                           'ValidationFrequency', floor(trainDs.NumObservations/optVars.minibatchSize), ...
                           'ValidationPatience', 20, ...
                           'MiniBatchSize', optVars.minibatchSize, ...
                           'ExecutionEnvironment', 'multi-gpu', ...
                           'OutputNetwork','best-validation-loss', ...
                           'Verbose', false);
        
        rockNet = trainNetwork(trainDs,lgraph,options2);
        close(findall(groot, 'Tag', 'NNET_CNN_TRAININGPLOT_UIFIGURE'))
        
        % Make and evaluate predictions
        predCat = classify(rockNet, valDs);
        obsCat = valTbl.categories;
        valError = 1 - mean(predCat == obsCat);
        
        % Save network
        fileName = [outputFolderName '\' num2str(valError) '.mat'];
        if ~exist(fileName, 'file')
            save(fileName, 'rockNet','valError','options1','options2')
        else
            % Prevent overwriting
            version = 1;
            fileName = ['(' num2str(version) ')' fileName];
            while exist(fileName, 'file')
                version = version + 1;
                fileName = ['(' num2str(version) ')' fileName(4:end)];
            end
        end
        
        % This output representing variable constraints is apparently
        % required by one of the above functions, but can be left blank.
        cons = [];
        
    end
end

%% Function to create FC layers
function layers = createFClayers(numLayers, numCats, numInputs)
    numLayers = numLayers + 1;
    layers = [];
    for i = 1:numLayers
        % If it's the last layer, should have units equal to the number of
        % categories. Otherwise, it should have units equal to the last
        % layer
        if i == numLayers
            layers = [layers
                fullyConnectedLayer(numCats, 'name', ['fcAppended' num2str(i)])
                softmaxLayer];
        else
            layers = [layers
                fullyConnectedLayer(numInputs, 'name', ['fcAppended' num2str(i)])
                reluLayer
                batchNormalizationLayer
                dropoutLayer(0.5)];
        end
    end
end

%% Function to freeze weights, stolen from the documentation
function layers = freezeWeights(layers)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

end

%% Function to unfreeze weights, modified from previous function
function layers = unfreezeWeights(layers)

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 1;
        end
    end
end

end

%% Function to reconnect the graph after freezing weights, again stolen from documentation
function lgraph = createLgraphUsingConnections(layers,connections)

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end

for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

end