function gmms = gmmTrain(dir_train, max_iter, epsilon, M)
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture
    gmms = {};    
    [mfccs, D] = getMfccs(dir_train);
    list_names = fieldnames(mfccs);
    index = 1;
    
    for s = list_names'
        theta = initTheta(mfccs, D, M);
        speaker = char(s);
        X = mfccs.(speaker);
        iter = 1;
        prev_L = -Inf;
        improv = Inf;
        while iter <= max_iter && improv >= epsilon
            [L, b] = computeLikelihood(X, theta, M);
            theta = updateParameters(theta, X, b, M);
            improv = L - prev_L;
            prev_L = L;
            iter = iter + 1;
        end
        
        gmms{index} = {};
        gmms{index}.name = speaker;
        gmms{index}.weights = theta.weights;
        gmms{index}.means = theta.means;
        gmms{index}.cov = theta.cov;
        
        index = index +1;
    end
    fprintf('The final likelihood is %s\n', L)
end



function theta=updateParameters(theta, X, b, M)
    D = size(X,2); % number of dimensions
    T = size(X, 1); % Number of Training Lines
    pmx_den = sum(repmat(theta.weights, T, 1).*b, 2);
    pmx = zeros(T,M);
    
    for m=1:M
       w = theta.weights(1,m);
       pmx(:,m) = repmat(w, T, 1).*b(:,m)./pmx_den;
    end
    
    %  Update Parameters
    theta.weights = sum(pmx,1)/T; % 1xM
    theta.means = (pmx' * X)'./repmat(sum(pmx, 1),D,1); % DxM
    new_covs = ((pmx' * (X.^2))'./repmat(sum(pmx, 1),D,1)) - theta.means.^2; % DxM
    
    for m=1:M
       theta.cov(:,:,m) = diag(new_covs(:,m)); 
    end
end


function theta = initTheta(mfccs, D, M)
    sp_names = fieldnames(mfccs);
    
    cov_mat = [];
    for m=1:M
        cov_mat(:,:,m) = eye(D);
    end
    
    theta = struct();
    theta.weights = 1/M * ones(1,M); %1xM
    theta.means = getMeans(mfccs, M); %DxM
    theta.cov = cov_mat; %DxDxM
end

function mean = getMeans(mfccs, M)
    mean = [];
    list_names = fieldnames(mfccs);
    num_sp = size(list_names,1);
    for i=1:M
        % get a random speaker
        rand_spkr = randperm (num_sp, 1);
        speaker = list_names{rand_spkr};
        % random mfcc vector for a random speaker
        mfcc_row = randperm (size(mfccs.(speaker), 1), 1);
        rand_vec = mfccs.(speaker)(mfcc_row,:);
        mean = [mean;rand_vec];
    end
    mean = mean';
end


function [mfccs, D] = getMfccs(dir_train)
    mfccs = struct();
    folders = dir(dir_train);
    for folder = folders'
        files = dir(fullfile(dir_train, folder.name, '*.mfcc'));
        speakerName = char(folder.name);
        
        if ~strcmp(speakerName,'.') && ~strcmp(speakerName,'..')
            if ~isfield(mfccs, speakerName)
               mfccs.(speakerName) = []; 
            end
            
            for file = files'
                mfcc_path = fullfile(dir_train, speakerName, file.name);
                mfccs.(speakerName) = vertcat(mfccs.(speakerName), dlmread(mfcc_path));    
            end
            D = size(mfccs.(speakerName), 2);
        end
    end

end
