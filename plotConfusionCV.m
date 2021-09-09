% This script computes the confusion matrices of the tested models

clear all
close all
warning off

GeneratePDF = 1;

thisGrid = 'Square';
models = {'Raw MobileNetV2','Raw GoogLeNet','Raw MobileNetV2+GoogLeNet','Shifted MobileNetV2','Shifted GoogLeNet','Shifted MobileNetV2+GoogLeNet'};
netname1 = 'mobilenetv2';
netname2 = 'googlenet';

numclasses = 7; %Number of classes of the trained model
maxShift = 11;
gridSpacing = 1;

accMeanGridTest = zeros(numel(models),10);

fGrid = figure(1);
CMGrid = cell(numel(models),10);
CMStats = cell(numel(models),10);
CMDetailedStats = cell(numel(models),10);
CMSimple = cell(numel(models),10);

for i = 1:10


    load(sprintf('models/p%i_%s_%i.mat',numclasses,netname2,i))
    fprintf('Testing %i classes model using split %i\n',numclasses,i)
    classes = net_train.Layers(end).Classes;
%     classes = categorical({'akiec';'bcc';'bkl';'df';'mel';'nv';'vasc'});

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

    % Mean scores
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
    
    
    numShifts = size(gridpred,2);
    % Combined shifted models
    clear gridpred gridscores confusion
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
    accMeanGridTest(6,i) = sum(classes(maxid) == testSet{i}.Labels)/numel(testSet{i}.Labels);
    
    
    % Confusion matrix calculation
    [~, ~, Glabels] = unique(cellstr(testSet{i}.Labels));
    [~, ~, Gpred] = unique(cellstr(pred1'));
    [CMGrid{1,i},CMStats{1,i},CMDetailedStats{1,i}]= confusion.getMatrix(Glabels,Gpred,0);
    [~, ~, Gpred] = unique(cellstr(pred2'));
    [CMGrid{2,i},CMStats{2,i},CMDetailedStats{2,i}]= confusion.getMatrix(Glabels,Gpred,0);
    [~, ~, Gpred] = unique(cellstr(classes(maxidTwoRaws)));
    [CMGrid{3,i},CMStats{3,i},CMDetailedStats{3,i}]= confusion.getMatrix(Glabels,Gpred,0);
    [~, ~, Gpred] = unique(cellstr(classes(maxid1)));
    [CMGrid{4,i},CMStats{4,i},CMDetailedStats{4,i}]= confusion.getMatrix(Glabels,Gpred,0);
    [~, ~, Gpred] = unique(cellstr(classes(maxid2)));
    [CMGrid{5,i},CMStats{5,i},CMDetailedStats{5,i}]= confusion.getMatrix(Glabels,Gpred,0);
    [~, ~, Gpred] = unique(cellstr(classes(maxid)));
    [CMGrid{6,i},CMStats{6,i},CMDetailedStats{6,i}]= confusion.getMatrix(Glabels,Gpred,0);


    figure(fGrid)
    fsGrid = subplot(2,5,i);
    cm = confusionchart(CMGrid{5,i},classes, ...
                'FontSize',14);
    sortClasses(cm,{'akiec','bcc','bkl','df','mel','nv','vasc'});
    title(sprintf('Split %i',i))  
    
end

if GeneratePDF 
    PdfFileName=sprintf('./plots/Confusion_c%i_%s_%sGrid_MeanScores',numclasses,'twoNets',thisGrid);
    set(gcf,'PaperUnits','centimeters');
    set(gcf,'PaperOrientation','portrait');
    set(gcf,'PaperPositionMode','manual');
    set(gcf,'PaperSize',[52 18]);
    set(gcf,'PaperPosition',[-4 0 60 18]);
    set(gca,'fontsize',14);
    saveas(gcf,PdfFileName,'pdf');
end


fModels = figure(2);
for NdxModel = 1:numel(models)
    
    figure(fModels)
    fsModels = subplot(1,6,NdxModel);
    AllCMGrid = cat(3, CMGrid{NdxModel,:});
    format short g
    disp('Mean and std CM Grid')
    MeanCMGrid = mean(AllCMGrid,3);
    StdCMGrid = std(AllCMGrid,[],3);
    disp(MeanCMGrid)
    disp(StdCMGrid)
    cm = confusionchart(round(MeanCMGrid),classes, ...
        'FontSize',22);
    sortClasses(cm,{'akiec','bcc','bkl','df','mel','nv','vasc'});
    title(sprintf('%s',models{NdxModel}))
    
end
if GeneratePDF 
    PdfFileName=sprintf('./plots/ComparisonConfusion_c%i_%s_%sGrid_MeanScores.pdf',...
        numclasses,'twoNets',thisGrid);
    set(gcf,'PaperUnits','centimeters');
    set(gcf,'PaperOrientation','portrait');
    set(gcf,'PaperPositionMode','manual');
    set(gcf,'PaperSize',[104 18]);
    set(gcf,'PaperPosition',[-8 0 120 18]);
    saveas(gcf,PdfFileName,'pdf');
end


save('StatsCV_MedianScores.mat','CMGrid','CMStats','CMDetailedStats')
   
