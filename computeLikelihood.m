function [L, b] = computeLikelihood(X, theta, M)
% Helper functions
% input: 
%       mfcc: concatenated training data for user TLxD (# training lines by # dimensions)
%       gmm: The appropriate struct for a given speaker's model
%
% output: 
%       L: log likelihood of data under model
    D = size(X,2); % number of dimensions
    T = size(X, 1); % Number of Training Lines
    
    % compute likelihood
    b = zeros(T,M);
    for m=1:M
        mu_trans = theta.means(:,m)';
        co = diag(theta.cov(:,:,m))';   
        numer = sum((((X-repmat(mu_trans, T, 1)).^2)./repmat(co,T,1)), 2);
        numer = exp(-1/2 * numer);
        denom = ((2*pi)^(D/2) * sqrt(prod(co)));

        b(:,m) = numer/denom;
    end
    
    pmx_den = sum(repmat(theta.weights, T, 1).*b, 2);
    L = sum(log(pmx_den));   
end