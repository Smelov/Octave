% File: EM_HMM.m
%
% Copyright (C) Daphne Koller, Stanford Univerity, 2012

function [P loglikelihood ClassProb PairProb] = EM_HMM(actionData, poseData, G, InitialClassProb, InitialPairProb, maxIter)

% INPUTS
% actionData: structure holding the actions as described in the PA
% poseData: N x 10 x 3 matrix, where N is number of poses in all actions
% G: graph parameterization as explained in PA description
% InitialClassProb: N x K matrix, initial allocation of the N poses to the K
%   states. InitialClassProb(i,j) is the probability that example i belongs
%   to state j.
%   This is described in more detail in the PA.
% InitialPairProb: V x K^2 matrix, where V is the total number of pose
%   transitions in all HMM action models, and K is the number of states.
%   This is described in more detail in the PA.
% maxIter: max number of iterations to run EM

% OUTPUTS
% P: structure holding the learned parameters as described in the PA
% loglikelihood: #(iterations run) x 1 vector of loglikelihoods stored for
%   each iteration
% ClassProb: N x K matrix of the conditional class probability of the N examples to the
%   K states in the final iteration. ClassProb(i,j) is the probability that
%   example i belongs to state j. This is described in more detail in the PA.
% PairProb: V x K^2 matrix, where V is the total number of pose transitions
%   in all HMM action models, and K is the number of states. This is
%   described in more detail in the PA.

% Initialize variables
N = size(poseData, 1);
K = size(InitialClassProb, 2);
L = size(actionData, 2); % number of actions
V = size(InitialPairProb, 1);

ClassProb = InitialClassProb;
PairProb = InitialPairProb;

loglikelihood = zeros(maxIter,1);

P.c = [];
P.clg.sigma_x = [];
P.clg.sigma_y = [];
P.clg.sigma_angle = [];

% EM algorithm
for iter=1:maxIter
  
  % M-STEP to estimate parameters for Gaussians
  % Fill in P.c, the initial state prior probability (NOT the class probability as in PA8 and EM_cluster.m)
  % Fill in P.clg for each body part and each class
  % Make sure to choose the right parameterization based on G(i,1)
  % Hint: This part should be similar to your work from PA8 and EM_cluster.m
  
  P.c = zeros(1,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
I = zeros(L,1);
for i = 1:L,
I(i) = actionData(1, i).marg_ind(1);
end
C = ClassProb(I, :);
P.c = sum(C)/L;

for k = 1:K,
Gshare = (length(size(G)) == 2);
if Gshare,
Ginuse = G;
else
Ginuse = reshape(G(:, :, k), 10, 2);
end

for i = 1:10,
if Ginuse(i, 1) ==0,
[P.clg(i).mu_y(k), P.clg(i).sigma_y(k)] = FitG(poseData(:, i, 1), ClassProb(:, k));
[P.clg(i).mu_x(k), P.clg(i).sigma_x(k)] = FitG(poseData(:, i, 2), ClassProb(:, k));
[P.clg(i).mu_angle(k), P.clg(i).sigma_angle(k)] = FitG(poseData(:, i, 3), ClassProb(:, k));
else
parent = Ginuse(i, 2);
parentdata = reshape(poseData(:,parent, :), N, 3);
[beta_y, P.clg(i).sigma_y(k)] = FitLG(poseData(:, i, 1), parentdata, ClassProb(:, k));
[beta_x, P.clg(i).sigma_x(k)] = FitLG(poseData(:, i, 2), parentdata, ClassProb(:, k));
[beta_a, P.clg(i).sigma_angle(k)] = FitLG(poseData(:, i, 3), parentdata, ClassProb(:, k));
P.clg(i).theta(k, :) = [beta_y(4);beta_y(1:3);beta_x(4);beta_x(1:3);beta_a(4);beta_a(1:3)];
end
end
end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % M-STEP to estimate parameters for transition matrix
  % Fill in P.transMatrix, the transition matrix for states
  % P.transMatrix(i,j) is the probability of transitioning from state i to state j
  P.transMatrix = zeros(K,K);
  
  % Add Dirichlet prior based on size of poseData to avoid 0 probabilities
  P.transMatrix = P.transMatrix + size(PairProb,1) * .05;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = reshape(sum(PairProb), K, K);
P.transMatrix = P.transMatrix + X;
for k = 1:K,
P.transMatrix(k, :) = P.transMatrix(k,:)/sum(P.transMatrix(k,:));
end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP preparation: compute the emission model factors (emission probabilities) in log space for each 
  % of the poses in all actions = log( P(Pose | State) )
  % Hint: This part should be similar to (but NOT the same as) your code in EM_cluster.m
  
  logEmissionProb = zeros(N,K);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
logClassProb = log(P.c);
logProb = repmat(logClassProb, N, 1)*0;
for n = 1:N,
data = reshape(poseData(n, :, :), 10, 3);
for k = 1:K,
if Gshare,
Ginuse = G;
else
Ginuse = reshape(G(:, :, k), 10, 2);
end
for i = 1:10,
if Ginuse(i, 1) == 0,
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 1), P.clg(i).mu_y(k), P.clg(i).sigma_y(k));
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 2), P.clg(i).mu_x(k), P.clg(i).sigma_x(k));
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 3), P.clg(i).mu_angle(k), P.clg(i).sigma_angle(k));
else
parent = [1 data(Ginuse(i, 2), :)];
mu_y = sum(parent .* P.clg(i).theta(k,1:4));
mu_x = sum(parent .* P.clg(i).theta(k,5:8));
mu_angle = sum(parent .* P.clg(i).theta(k,9:12));
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 1), mu_y, P.clg(i).sigma_y(k));
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 2), mu_x, P.clg(i).sigma_x(k));
logProb(n, k) = logProb(n, k) + lognormpdf(data(i, 3), mu_angle, P.clg(i).sigma_angle(k));
end
end
end
end
logEmissionProb = logProb;

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
    
  % E-STEP to compute expected sufficient statistics
  % ClassProb contains the conditional class probabilities for each pose in all actions
  % PairProb contains the expected sufficient statistics for the transition CPDs (pairwise transition probabilities)
  % Also compute log likelihood of dataset for this iteration
  % You should do inference and compute everything in log space, only converting to probability space at the end
  % Hint: You should use the logsumexp() function here to do probability normalization in log space to avoid numerical issues
  
  ClassProb = zeros(N,K);
  PairProb = zeros(V,K^2);
  loglikelihood(iter) = 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % YOUR CODE HERE
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for l = 1:L,
init = struct("var", [1], "card", [K], "val", log(P.c));
Acount = length(actionData(1, l).marg_ind);
factors = repmat(struct('var', [], 'card', [], 'val', []), Acount, 1);
for j = 1:Acount,
factors(j).var = j;
factors(j).card = K;
factors(j).val = logEmissionProb(actionData(1, l).marg_ind(j), :);
end
pairval = log(reshape(P.transMatrix, 1, K*K));
pairfactors= repmat(struct('var', [], 'card', [], 'val', []), Acount-1, 1);
for j = 1:Acount-1,
pairfactors(j).var = [j j+1];
pairfactors(j).card = [K K];
pairfactors(j).val = pairval;
end
allfactors = [init;factors;pairfactors];
[M, PCalibrated] = ComputeExactMarginalsHMM(allfactors);
for m=1:length(M),
Norm = logsumexp(M(m).val);
idx = actionData(1,l).marg_ind(M(m).var);
ClassProb(idx,:) = exp(M(m).val - Norm);
end
for c=1:length(PCalibrated.cliqueList),
idx = PCalibrated.cliqueList(c).var(1);
pidx = actionData(1,l).pair_ind(idx);
Norm = logsumexp(PCalibrated.cliqueList(c).val);
PairProb(pidx,:) = exp(PCalibrated.cliqueList(c).val - Norm);
end
loglikelihood(iter) = loglikelihood(iter) + Norm;
end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Print out loglikelihood
  disp(sprintf('EM iteration %d: log likelihood: %f', ...
    iter, loglikelihood(iter)));
  if exist('OCTAVE_VERSION')
    fflush(stdout);
  end
  
  % Check for overfitting by decreasing loglikelihood
  if iter > 1
    if loglikelihood(iter) < loglikelihood(iter-1)
      break;
    end
  end
  
end

% Remove iterations if we exited early
loglikelihood = loglikelihood(1:iter);
