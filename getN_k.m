function N_k = getN_k (k);
% N_K Calculates the number of samples whose label is k (1, 2, or 3)

global S_label

if k == 1
    N_k = length (S_label(S_label == 1));
elseif k == 2
    N_k = length (S_label(S_label == 2));
elseif k == 3
    N_k = length (S_label(S_label == 3));
end

end