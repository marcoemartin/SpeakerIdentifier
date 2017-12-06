addpath(genpath('/u/cs401/A3_ASR/code/FullBNT-1.0.7'));
dir_train = '/u/cs401/speechdata/Training';

M = 4;
Q = 3;
D = 7;
max_iter = 5;
phnmDict = {};

folders = dir(dir_train);
for folder = folders'
    phnfiles = dir(fullfile(dir_train, folder.name, '/*.phn'));
    speakerName = char(folder.name);

    if ~strcmp(speakerName,'.') && ~strcmp(speakerName,'..')
        for file = phnfiles'
            mfccName = [file.name(1:end-3), 'mfcc'];
            
            phnPath = fullfile(dir_train, speakerName, file.name);
            [starts, ends, Phns] = textread(phnPath, '%d %d %s', 'delimiter','\n');
            
            matrix = load([dir_train, filesep, speakerName, filesep, mfccName]);
            matrix = matrix';
            matrix = matrix(1:D, :);
            
            for idx = 1:length(Phns)
                Start = starts(idx)/128 +1;
                End = min(ends(idx)/128 +1, length(matrix));
                phn = char(Phns(idx));
                if strcmp(phn, 'h#')
                    phn = 'sil';
                end
                
                if ~isfield(phnmDict, phn)
                    phnmDict.(phn) = {};
                end
                phnmDict.(phn){length(phnmDict.(phn))+1} = matrix(:, Start:End);
            end
        end
    end
end


HMM = struct();
fields = fieldnames(phnmDict);
for i = 1:length(fields)
    phn = fields{i};
    HMM.(phn) = initHMM(phnmDict.(phn), M, Q);
    [HMM.(phn), LL] = trainHMM(HMM.(phn), phnmDict.(phn), max_iter);
end

save('savedHMM.mat', 'HMM', '-mat');