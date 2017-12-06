dir_test = 'AccuracyData/';
%dir_test = '/u/cs401/speechdata/Testing';
dir_train = '/u/cs401/speechdata/Training';
M = 4;
epsilon = 1;
D = 5;
iter = 5;

prefix = 'identifiedSpeakers/';
suffix = 'lik';

total = 0;
correct = 0;
wrong = 0;


fileID = fopen('answers.txt','r');
formatSpec = '%s';
A = textscan(fileID,formatSpec);

gmms = gmmTrain(dir_train, iter, epsilon, M);

mfccfiles = dir(fullfile(dir_test, '/*.mfcc'));
ll = zeros(1, length(mfccfiles));

mkdir(prefix(1:end-1));
indx = 1;

for file = mfccfiles'
    mfccName = file.name;
    X = load([dir_test, filesep, mfccName]); %Unknown speaker matrix
    D = size(X, 2); % number of dimensions
    T = size(X, 1); % Number of Training Lines
    
    for s=1:length(gmms)
        theta = struct();
        theta.means = gmms{s}.means;
        theta.weights = gmms{s}.weights;
        theta.cov = gmms{s}.cov;
        
        
        %compute likeliness of unkown speakers matrix with current speaker
        [L, b] = computeLikelihood(X, theta, M);
        ll(1,s) = L;
    end
    
    [sortedL,sortingIndices] = sort(ll,'descend');
    maxValues = sortedL(1:5);
    indxVals = sortingIndices(1:5);
    
    %Speaker names
    top1 = gmms{indxVals(1)}.name;
    top2 = gmms{indxVals(2)}.name;    
    top3 = gmms{indxVals(3)}.name;
    top4 = gmms{indxVals(4)}.name;
    top5 = gmms{indxVals(5)}.name;
    
    %Liklihoods
    ll1 = maxValues(1);
    ll2 = maxValues(2);
    ll3 = maxValues(3);
    ll4 = maxValues(4);
    ll5 = maxValues(5);

    fname = strcat(prefix, mfccName(1:end-4), suffix);
    output = fopen(fname, 'wt');
    fprintf(output, '%s %d\n%s %d\n%s %d\n%s %d\n%s %d\n', top1,ll1, top2,ll2, top3,ll3, top4,ll4, top5,ll5);
    fclose(output);
    
    % Check accuracy
    if strcmp(mfccName(1:end-5), top1) || strcmp(mfccName(1:end-5), top2)
       correct = correct + 1; 
    else
        wrong = wrong + 1;
    end
    total = total +1;
end

fprintf('Accuracy %d/%d\n', correct, total)
    
