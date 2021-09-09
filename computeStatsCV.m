% This script computes the statistics of the 10-fols CV
clear all
warning off

grids = {'Square'};
netname1 = 'mobilenetv2';
netname2 = 'googlenet';

numclasses = 7; %Number of classes of the trained model
maxShift = 11;
gridSpacing = 1;

accModeGridTest = zeros(numel(grids),10);
accMaxGridTest = zeros(numel(grids),10);
accMeanGridTest = zeros(numel(grids),10);
accMedianGridTest = zeros(numel(grids),10);
accSimpleTest = zeros(1,10);

for i = 1:10

    load(sprintf('models/p%i_%s_%i.mat',numclasses,netname1,i))
    fprintf('Testing %i classes model using split %i\n',numclasses,i)
    classes = net_train.Layers(end).Classes;

    if numclasses == 2
        ii = testSet{i}.Labels ~= 'nv';
        testSet{i}.Labels(ii) = 'mel';
        testSet{i}.Labels = removecats(testSet{i}.Labels);
    end

    numImages = length(testSet{i}.Files);

    for NdxGrid = 1:numel(grids)

        thisGrid = grids{NdxGrid};
        
        load(sprintf('results/c%i_%s_%i_%sGrid_max%i.mat', numclasses, netname1, i, thisGrid, maxShift))
        gridpred1 = gridpred;
        gridscores1 = gridscores;
        pred1 = pred;
        
        load(sprintf('results/c%i_%s_%i_%sGrid_max%i.mat', numclasses, netname2, i, thisGrid, maxShift))
        gridpred2 = gridpred;
        gridscores2 = gridscores;
        pred2 = pred;
        
        numShifts = size(gridpred,2);
        clear gridpred gridscores
        gridpred = categorical.empty(numImages,0);
        gridscores = cell(numImages,numShifts);
        idgr = randperm(numShifts);
        idgr1 = idgr(1:numel(idgr)/2); idgr2 = idgr(numel(idgr)/2+1:end);
        gridpred(:,idgr1) = gridpred1(:,idgr1);
        gridpred(:,idgr2) = gridpred2(:,idgr2);
        gridscores(:,idgr1) = gridscores1(:,idgr1);
        gridscores(:,idgr2) = gridscores2(:,idgr2);
        
        
        % Accuracy calculation

        % Mode
        y = mode(gridpred,2);
        accModeGridTest(NdxGrid,i) = sum(y == testSet{i}.Labels)/numel(testSet{i}.Labels);

        % Max, mean and median scores
        MaxScores = zeros(numImages,numclasses);
        MeanScores = zeros(numImages,numclasses);
        MedianScores = zeros(numImages,numclasses);
        for NdxIm = 1:numImages

            MaxScores(NdxIm,:) = max(cell2mat(gridscores(NdxIm,:)'),[],1);
            MeanScores(NdxIm,:) = mean(cell2mat(gridscores(NdxIm,:)'),1);
            MedianScores(NdxIm,:) = median(cell2mat(gridscores(NdxIm,:)'),1);

        end
        [~,maxid] = max(MaxScores,[],2);
        accMaxGridTest(NdxGrid,i) = sum(classes(maxid) == testSet{i}.Labels)/numel(testSet{i}.Labels);
        [~,maxid2] = max(MeanScores,[],2);
        accMeanGridTest(NdxGrid,i) = sum(classes(maxid2) == testSet{i}.Labels)/numel(testSet{i}.Labels);
        [~,maxid3] = max(MedianScores,[],2);
        accMedianGridTest(NdxGrid,i) = sum(classes(maxid3) == testSet{i}.Labels)/numel(testSet{i}.Labels);

        % Printing results
        fprintf('Accuracy with %s grid using [mode max mean median]: [%f %f %f %f]\n',thisGrid,...
            accModeGridTest(NdxGrid,i),accMaxGridTest(NdxGrid,i),accMeanGridTest(NdxGrid,i),accMedianGridTest(NdxGrid,i));

    end

    accSimpleTest(i) = sum(pred1' == testSet{i}.Labels)/numel(testSet{i}.Labels);
    fprintf('Accuracy with simple testing of %s: %f\n',netname1,accSimpleTest(i));
    accSimpleTest(i) = sum(pred2' == testSet{i}.Labels)/numel(testSet{i}.Labels);
    fprintf('Accuracy with simple testing of %s: %f\n',netname2,accSimpleTest(i));    
end

fprintf('\n')
fprintf('Mean and std accuracy using mode scores: %f %f %f %f %f +- %f %f %f %f %f \n', ...
    mean(accModeGridTest,2), std(accModeGridTest,[],2));
fprintf('Mean and std accuracy using max scores: %f %f %f %f %f +- %f %f %f %f %f \n', ...
    mean(accMaxGridTest,2), std(accMaxGridTest,[],2));
fprintf('Mean and std accuracy using mean scores: %f %f %f %f %f +- %f %f %f %f %f \n', ...
    mean(accMeanGridTest,2), std(accMeanGridTest,[],2));
fprintf('Mean and std accuracy using median scores: %f %f %f %f %f +- %f %f %f %f %f \n', ...
    mean(accMedianGridTest,2), std(accMedianGridTest,[],2));
fprintf('Mean and std accuracy with simple testing: %f +- %f \n', ...
    mean(accSimpleTest,2), std(accSimpleTest,[],2));

   
