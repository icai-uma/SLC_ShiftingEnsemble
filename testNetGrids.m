% Code for testing a 10-fold CV using the shifting technique
clear all

% Change this path to point where the HAM10000 folder is
dataset_rootpath = 'G:\Unidades compartidas\ICAI\datasets';

grids = {'Square'};
netname = 'googlenet';
% netname = 'mobilenetv2';

numclasses = 7;
maxShift = 11;
gridSpacing = 1;

for i = 1:10

    load(sprintf('models/p%i_%s_%i.mat',numclasses,netname,i))
    fprintf('Testing %i classes %s model using split %i\n',numclasses,netname,i)
    
    if numclasses == 2
        ii = testSet{i}.Labels ~= 'nv';
        testSet{i}.Labels(ii) = 'mel';
        testSet{i}.Labels = removecats(testSet{i}.Labels);
    end

    numImages = length(testSet{i}.Files);

    for NdxGrid = 1:numel(grids)

        thisGrid = grids{NdxGrid};
        fprintf('Testing using %s grid...\n',thisGrid);
        gr = createGrid(thisGrid,maxShift,gridSpacing);
        numShifts = size(gr,2);

        pred = categorical.empty(numImages,0);
        gridpred = categorical.empty(numImages,0);
        gridscores = cell(numImages,numShifts);

        for NdxIm=1:numImages

            impath = testSet{i}.Files{NdxIm};
            if ~exist(impath,'file')
                impath = strrep(impath,'\\home\enriqued\Documents\MATLAB\Melanoma',dataset_rootpath);
            end
            im = imread(impath);
            im = imresize(im,net.Layers(1).InputSize(1:2));

            % Testing
            tic
            pred(NdxIm) = net_train.classify(im);
            for NdxShift=1:numShifts
                shiftedIm = circshift(im,gr(:,NdxShift)');
                [gridpred(NdxIm,NdxShift), gridscores{NdxIm,NdxShift}] = net_train.classify(shiftedIm);
            end
            timeSimTest = toc;
            fprintf('Image %i of %i processed in %d seconds\n',NdxIm,numImages,timeSimTest);

        end

        save(sprintf('results/c%i_%s_%i_%sGrid_max%i_sp%i', numclasses, netname, i, grids{NdxGrid}, maxShift, gridSpacing),...
            'pred','gr','gridpred','gridscores')

    end
    
end

