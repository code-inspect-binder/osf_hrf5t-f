% Template script to create ResNet50-based scaling solution. Assumes a
% trained network is available. File paths and network layer are left
% blank, to be filled in by the user.
% 
% Date: 8/14/22
% Author: Brian J. Meagher

%% Load the relevant data
nstimpersubtype = 18;

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

% Create datastores
imds = imageDatastore({image_location_standard, image_location_photoshopped});
aug = augmentedImageDatastore([224 224], imds);

% Rock names for plots
rockNames = ["Andesite", "Basalt", "Diorite", "Gabbro", "Granite", ...
             "Obsidian", "Pegmatite", "Peridotite", "Pumice", "Rhyolite", ...
            "Amphibolite", "Anthracite", "Gneiss", "Hornfels", "Marble", ...
            "Migmatite", "Phyllite", "Quartzite", "Schist", "Slate", ...
            "Bituminous Coal", "Breccia", "Chert", "Conglomerate", ...
            "Dolomite", "Micrite", "Rock Gypsum", "Rock Salt", ...
            "Sandstone", "Shale"];

% Original MDS dimensions
std = load([]); % Insert path to rocksmds480.dat here
hsn = load([]); % Insert path to rocksmdsphoto60.dat here

mds = [std; hsn(:,1:8)];
        
%% Get activations
load([]); % Insert path to network .mat file here.
net = rockNet;

% Find and extract activations of second-to-last fully-connected layer
layerNames = arrayfun(@(x) x.Name, net.Layers, 'UniformOutput', false);
lastFC = layerNames{min(find(contains(layerNames, 'fc'),2,'last'))};
activRecord = activations(net, aug, lastFC); 

% Reshape the array
activRecord = permute(activRecord,[4 3 1 2]);

%% Create initial scaling solution

% Calculate distance between each stimulus, treating activations as
% coordinates
D = squareform(pdist(activRecord));

% Scale the data
opts = statset('UseParallel', true);
[Y, stress] = mdscale(D, 8, 'Options', opts, 'Replicates', 5, 'Start', 'random', 'criterion', 'stress');

% Procrustes analysis to rotate 8d solution to be as close as possible to
% original solution. Keep original scale.
[~, rotatedMDS] = procrustes(mds, Y, 'scale', false);

%% Produce a heatmap

% Get coordinates
subD = squareform(pdist(Y));

% Rearrange into appropriate subtypes
swapIx = zeros(length(subD),1);
k = 0;
for i = 1:length(subD)
    j = mod(i,nstimpersubtype);
    if j ~= nstimpersubtype - 1 && j ~= 0
        k = k + 1;
        swapIx(i) = k;
    else
        typePad = floor((i-1)/nstimpersubtype);
        swapIx(i) = numRocksStandard + typePad*2 + ((j == 0) + 1);
    end
end

subD = subD(swapIx,:); subD = subD(:,swapIx);

% Compute average distances
subAverages = zeros(30);
for i = 1:length(subAverages)
    for j = 1:length(subAverages)
        ixMin = (i-1)*nstimpersubtype + 1;
        ixMax = ixMin + nstimpersubtype - 1;
        jxMin = (j-1)*nstimpersubtype + 1;
        jxMax = jxMin + nstimpersubtype - 1;

        subMatrix = subD(ixMin:ixMax,jxMin:jxMax);
        subMatrix = subMatrix(:);
        subAverages(i,j) = mean(subMatrix(subMatrix ~= 0));
    end
end

f = figure;
f.Position(3:4) = [1600 900];
heatmap(rockNames, rockNames, subAverages);

exportgraphics(gcf, 'distanceHeatmap.png');

%% Plot the images on pairs of dimensions
scale = 0.75;
dimScale = 1e4;

D = rotatedMDS;

for ds = 1:size(D,2)/2
    dimScaleY = dimScale/range(D(:,(ds*2)));
    dimScaleX = dimScale/range(D(:,(ds*2)-1));

    f = figure;
    hold on

    for i = 1:540
        [img, ~, alpha] = imread(names{i});
        img = imresize(img, scale);
        alpha = imresize(alpha, scale);
        
        y = dimScaleY*(D(i,(ds*2)));
        x = dimScaleX*(D(i,(ds*2)-1));
        image(img, 'XData', x, 'YData', y, 'AlphaData', alpha, 'AlphaDataMapping', 'direct');
    end
    hold off

    f.Position(3:4) = [800 800];
    set(gca,'visible','off')
end

%% Determine correlations between CNN-derived and similarity-based dimensions
diag(corr(rotatedMDS, mds))

%% Write solutions to files
fileName = 'deepLearningMDS_rotated.txt';
fid = fopen(fileName, 'w');
formatSpec = [repmat('%8.10f ',1,8) '\n'];
fprintf(fid, formatSpec, rotatedMDS);
fclose('all');