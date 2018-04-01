% It's an implementation of the Bagging - Bootstrap AGGregatING algorithm from BREIMAN, 1996.
%
% The training an classification stages are run. It's tested using an 10-fold cross validation process.
%
% $Author: Derzu Omaia

close all;

% remove prtools warnings
prwarning(0);

% teste index
indiceDB = 6;
        
DBS_name = [{'data/UCI/balance-scale/balance-scale.data.txt'} ;...%01  % 3 different classes 
            {'data/UCI/bcw/breast-cancer-wisconsin.data.txt'} ;...%02 
            {'data/UCI/musk/clean2.data'} ;...%03
            {'data/UCI/glass/glass.data.txt'} ;...%04 
            {'ionosphere.mat'} ;...%05 
            {'iris.dat'} ;...%06
            {'data/UCI/mfeat/mfeat-fac.txt'} ;...%07
            {'data/UCI/mfeat/mfeat-kar.txt'} ;...%08
            {'data/UCI/mfeat/mfeat-mor.txt'} ;...%09
            {'data/UCI/mfeat/mfeat-zer.txt'} ;...%10
            {'data/UCI/optdigits/optdigits.all'} ;...%11
            {'data/UCI/pendigits/pendigits.all'} ;...%12
            {'data/UCI/diabetes/pima-indians-diabetes.data.txt'} ;...%13
            {'data/UCI/segmentation/segmentation.data.data'} ;...%14 % 7 different classes 
            {'data/UCI/segmentation/segmentation.test.data'} ;...%15
            {'data/UCI/sonar/sonar.all-data.txt'} ;...%16 
            {'data/UCI/vowel/vowel-context.data.txt'} ;...%17  % 10 different classes
            ];

[dadosX, dadosY] = readDB(indiceDB, DBS_name);

fprintf('Testing DB %s\n', DBS_name{indiceDB});
Nfolds = 10;
indices = crossvalind('Kfold', size(dadosX, 1), Nfolds);

% NUMBER OF CLASSIFIERS INSIDE THE ENSEMBLE.
% L=10; %% number of ensemble individuals;
% try diferentes L's
Lini = 10;
Lp   = 30;
Lfim = 100;

if mod(Lfim-Lini,Lp)==0
    d=floor((Lfim-Lini)/Lp+1);
else
    d=floor((Lfim-Lini)/Lp);
end

accBagg_m=zeros(1,d);
acc_st_m = 0;

for Kfold=1:Nfolds
    % cross validation, separa os 2 grupos de treino e teste.
    test = (indices == Kfold); train = ~test;
    testX = dadosX(test, :); % testX is de test set
    testY = dadosY(test, :); % Y vector is a label vector
    trainX = dadosX(train, :); % trainX is the training set
    trainY = dadosY(train, :); % Y vector is a label vector
    
    % create just one clissifier, trained without bagging, just for comparing the with bagging.
    single_tree = geraDecisionTree(trainX, trainY, 0);
    acc_st = 1-classificaUm(single_tree, testX, testY);
    acc_st_m = acc_st_m + acc_st;
    
    numberfeature=size(trainX,2);

    accBagg = zeros(1, d);

    fprintf('Kfold=%d ', Kfold);

    for L=Lini:Lp:Lfim
        % Create and classify a pool of classifing using bagging.
        pool = baggingPool(trainX, trainY, L);
        accBagg(floor((L-Lini)/Lp+1)) = 1-classificaPoolSimple(pool, testX, testY);
        accBagg_m(floor((L-Lini)/Lp+1)) = accBagg_m(floor((L-Lini)/Lp+1)) + accBagg(floor((L-Lini)/Lp+1));
    end
end % Kfold

%Faz a media
% Avarage accurace
accBagg_m = accBagg_m./Nfolds;
acc_st_m = acc_st_m./Nfolds;

fprintf('\n%s:\n', DBS_name{indiceDB});
fprintf('single tree = %f\n', acc_st_m );

accBagg_m = [accBagg_m ; zeros(1, d)];

for L=Lini:Lp:Lfim
    fprintf('L=%3d', L);
    fprintf(' bagging = %f\n', accBagg_m(1, floor((L-Lini)/Lp+1)) );
    accBagg_m(2 , floor((L-Lini)/Lp+1) ) = acc_st_m;
end

displayChartSingle( accBagg_m, Lini, Lp, Lfim, 'single tree', 'bagging');