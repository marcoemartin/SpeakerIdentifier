dir_test = '/u/cs401/speechdata/Testing';

fn_output = 'recogAccuracy.txt';
fn_HMM = 'savedHMM.mat';

output = fopen(fn_output, 'w');

total = 0;
correct = 0;
wrong = 0;

D = 7;

load(fn_HMM);
phnfiles = dir(fullfile(dir_test, '/*.phn'));


for file = phnfiles'
    mfccName = [file.name(1:end-3), 'mfcc']; %Edit file name to get the corresponding Mfcc
    phnPath = fullfile(dir_test, file.name);
    [starts, ends, Phns] = textread(phnPath, '%d %d %s', 'delimiter','\n');

    % Get mfcc matrix of testing phoneme
    matrix = load([dir_test, filesep, mfccName]);
    matrix = matrix';
    matrix = matrix(1:D, :);

    % Loop through testing phonemes
    for idx = 1:length(Phns)
        Start = starts(idx)/128 +1;
        End = min(ends(idx)/128 +1, length(matrix));
        phn = char(Phns(idx));
        if strcmp(phn, 'h#')
            phn = 'sil';
        end
        
        hmm_phnms = fieldnames(HMM);
        testing = {};
        testing.phnm = '';
        testing.likelihood = -Inf;

        % Find largest likelihood of a testing phoneme given the 
        % mfcc matrix of the testing phoneme and the HMM to compare
        % it with 

        % Predict what phoneme this matrix belongs too 
        for p=1:length(hmm_phnms)
            %Compute the likelihood of this phoneme matrix 
            learned_phnm = char(hmm_phnms{p});
            testing_l = loglikHMM(HMM.(learned_phnm), matrix(:, Start:End));

            % Choose the phoneme's matrix with the highest
            % likelihood
            if testing_l > testing.likelihood
                testing.phnm = learned_phnm;
                testing.likelihood = testing_l;
            end
        end

        if strcmp(testing.phnm, phn)
           correct = correct +1; 
           result = ['Correctly guessed ', phn];
        else
           wrong = wrong +1; 
           result = ['Incorrectly guessed ', phn, ' as ', testing.phnm];
        end

        total = total +1;
        fprintf(output, '%s\n', result);
        disp(result)
    end
end


percent = (correct*100)/total;
accuracy = ['accuracy is: ', int2str(percent),'%'];
fprintf(output, '%s\n', accuracy);

fclose(output);
