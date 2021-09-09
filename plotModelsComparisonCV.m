% This script plots the graph bar comparing all models
clear all
warning off

thisGrid = 'Square';
models = {'Raw MobileNetV2','Raw GoogLeNet','Raw MobileNetV2+GoogLeNet','Shifted MobileNetV2','Shifted GoogLeNet','Shifted MobileNetV2+GoogLeNet'};
netname1 = 'mobilenetv2';
netname2 = 'googlenet';

numclasses = 7; %Number of classes of the trained model
maxShift = 11;
gridSpacing = 1;
numRuns = 30;

accMeanGridTest = zeros(numel(models),10);
accTwoNetsRuns = zeros(numRuns,10);

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

    % Load first net results
    load(sprintf('results/c%i_%s_%i_%sGrid_max%i_sp%i.mat', numclasses, netname1, i, thisGrid, maxShift, gridSpacing))
    gridpred1 = gridpred;
    gridscores1 = gridscores;
    pred1 = pred;

    % Load second net results
    load(sprintf('results/c%i_%s_%i_%sGrid_max%i_sp%i.mat', numclasses, netname2, i, thisGrid, maxShift, gridSpacing))
    gridpred2 = gridpred;
    gridscores2 = gridscores;
    pred2 = pred;
    
    % Load both nets results without shifting to compare
    load(sprintf('results/c%i_%s_%i.mat', numclasses, netname1, i))
    predscores1 = predscores;
    load(sprintf('results/c%i_%s_%i.mat', numclasses, netname2, i))
    predscores2 = predscores;


    % Accuracy calculation
    % Plain models
    accMeanGridTest(1,i) = sum(pred1' == testSet{i}.Labels)/numel(testSet{i}.Labels);
    accMeanGridTest(2,i) = sum(pred2' == testSet{i}.Labels)/numel(testSet{i}.Labels);
    
    % Combined plain models
    MeanScoresTwoRaws = mean(cat(3,predscores1,predscores2),3); % Change by median if needed
    [~,maxidTwoRaws] = max(MeanScoresTwoRaws,[],2);
    accMeanGridTest(3,i) = sum(classes(maxidTwoRaws) == testSet{i}.Labels)/numel(testSet{i}.Labels);

    % Shifted models
    MeanScores1 = zeros(numImages,numclasses);
    MeanScores2 = zeros(numImages,numclasses);
    for NdxIm = 1:numImages
        MeanScores1(NdxIm,:) = mean(cell2mat(gridscores1(NdxIm,:)'),1); % Change by median if needed
        MeanScores2(NdxIm,:) = mean(cell2mat(gridscores2(NdxIm,:)'),1); % Change by median if needed
    end
    [~,maxid1] = max(MeanScores1,[],2);
    accMeanGridTest(4,i) = sum(classes(maxid1) == testSet{i}.Labels)/numel(testSet{i}.Labels);
    [~,maxid2] = max(MeanScores2,[],2);
    accMeanGridTest(5,i) = sum(classes(maxid2) == testSet{i}.Labels)/numel(testSet{i}.Labels);
    
    % Combined shifted models
    numShifts = size(gridpred,2);
    for NdxRun = 1:numRuns
        
        rng(NdxRun)
        clear gridpred gridscores
        gridscores = cell(numImages,numShifts);
        idgr = randperm(numShifts);
        idgr1 = idgr(1:numel(idgr)/2); idgr2 = idgr(numel(idgr)/2+1:end);
        gridscores(:,idgr1) = gridscores1(:,idgr1);
        gridscores(:,idgr2) = gridscores2(:,idgr2);
        MeanScores = zeros(numImages,numclasses);
        for NdxIm = 1:numImages
            MeanScores(NdxIm,:) = mean(cell2mat(gridscores(NdxIm,:)'),1); % Change by median if needed
        end
       
        [~,maxid] = max(MeanScores,[],2);
        accTwoNetsRuns(NdxRun,i) = sum(classes(maxid) == testSet{i}.Labels)/numel(testSet{i}.Labels);
        
    end
    accMeanGridTest(6,i) = mean(accTwoNetsRuns(:,i));
      
end

% Plor bar plot (each row represents a group)
figure('WindowState', 'maximized')
b = bar(accMeanGridTest', 'grouped');
hold on
% Calculate the number of bars in each group
nbars = size(accMeanGridTest', 2);
% Get the x coordinate of the bars
x = b(nbars).XEndPoints;
% Plot the errorbars
errorbar(x',accMeanGridTest(6,:),std(accTwoNetsRuns,[],1),'k','linestyle','none')
hold off
ylim([0.76 0.86])
legend(models,'Orientation','horizontal','Location','bestoutside')
xlabel('CV Splits')
ylabel('Accuracy')
set(gca,'fontsize',16);
exportgraphics(gcf,sprintf('ModelsComparisonCV_MeanScores_%s.pdf',thisGrid),'BackgroundColor','none','ContentType','vector')


   
