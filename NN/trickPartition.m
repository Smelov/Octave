function p = trickPartition(rbm_w)
n = size(rbm_w,1)
allp = zeros(1, 2^size(rbm_w,1));
for i =1:size(allp, 2),
k = dec2bin(i-1);
l = logical(double(char(k))-48);
if length(l) < n,
l = [logical(zeros(1, n-length(l))) l];
end
aRow = sum(rbm_w(l, :), 1);
allp(i) = log(prod(1+exp(aRow)));
end
p = log(sum(exp(allp)))
end