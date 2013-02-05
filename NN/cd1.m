function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
% error('not yet implemented');
visible_data = sample_bernoulli(visible_data);
hidden_state_p = visible_state_to_hidden_probabilities(rbm_w, visible_data);
hidden_state_b = sample_bernoulli(hidden_state_p);
re_visible_state = sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w, hidden_state_b));
re_hidden_state = visible_state_to_hidden_probabilities(rbm_w, re_visible_state);
ret = configuration_goodness_gradient(visible_data, hidden_state_b) - configuration_goodness_gradient(re_visible_state, re_hidden_state);
end
