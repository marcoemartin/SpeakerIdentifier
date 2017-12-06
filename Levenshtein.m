function [SE IE DE LEV_DIST] =Levenshtein(hypothesis,annotation_dir)
% Input:
%	hypothesis: The path to file containing the the recognition hypotheses
%	annotation_dir: The path to directory containing the annotations
%			(Ex. the Testing dir containing all the *.txt files)
% Outputs:
%	SE: proportion of substitution errors over all the hypotheses
%	IE: proportion of insertion errors over all the hypotheses
%	DE: proportion of deletion errors over all the hypotheses
%	LEV_DIST: proportion of overall error in all hypotheses
UP = 0;
LEFT = 1;
UPLEFT = 2;

[Starts, Ends, Sents] = textread(hypothesis, '%d %d %s', 'delimiter', '\n');
prefix = 'unkn_';

SE = 0;
IE = 0;
DE = 0;
LEV_DIST = 0;
num_words = 0;

for s = 1:length(Sents)
    ref = [annotation_dir, prefix, int2str(s), '.txt'];
    disp(ref)
    [SA, EA, SenA] = textread(ref, '%d %d %s', 'delimiter', '\n');
    ref_sent = char(SenA(1));
	hyp_sent = char(Sents(s));
    %split on words
	ref_sent = strread(ref_sent,'%s','delimiter', ' ');
	hyp_sent = strread(hyp_sent, '%s', 'delimiter', ' ');

	n = length(ref_sent);
	m = length(hyp_sent);

	R = zeros(n+1, m+1);
	R(:,:) = Inf;
	R(1, 1) = 0;
    B = zeros(n+1, m+1);
    
    for i=2:n+1
        for j=2:m+1
			del = R(i-1, j)+1;
			sub = R(i-1, j-1) + ~strcmp(ref_sent(i-1), hyp_sent(j-1));
			ins = R(i, j-1) + 1;
			R(i, j) = min(del, min(sub, ins));
			if R(i, j) == del
				B(i, j) = UP;
			elseif R(i,j) == ins
				B(i,j) = LEFT;
			else
				B(i,j) = UPLEFT;
			end
        end
    end
    
    
	i = n+1;
	j = m+1;
	subs_repl = 0;
	ins_repl = 0;
	del_repl = 0;
	while i>1 && j>1
		if B(i,j) == UP
			del_repl = del_repl + 1;
			i = i -1;
		elseif B(i,j) == LEFT
			ins_repl = ins_repl + 1;
			j = j - 1;
		else
			if ~strcmp(ref_sent(i-1), hyp_sent(j-1))
				subs_repl = subs_repl + 1;
			end
			i = i-1;
			j = j-1;
		end
	end

	SE = SE + subs_repl;
	DE = DE + del_repl;
	IE = IE + ins_repl;
	num_words = num_words + length(ref_sent);
end

SE = SE/num_words;
DE = DE/num_words;
IE = IE/num_words;
LEV_DIST = SE + DE + IE;
end
