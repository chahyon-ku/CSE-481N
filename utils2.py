def make_mask(sent, tokens, total_sent_length, tokenizer=None):
    sent_len = len(tokens)
    mask_ids = list(range(total_sent_length))
    orig_tokens = sent.split()
    j = 0
    for t in orig_tokens:
        exp_tokens = tokenizer.tokenize(t)
        old_j = j
        for i, exp_token in enumerate(exp_tokens):
            T = tokens[j]
            assert T == exp_token
            j += 1

        #        lt = len(t)
        #        curr_len = 0
        #        curr_token = ''
        #        old_j = j
        #        exp_tokens = tokenizer.tokenize(t)
        #        lt = len(exp_tokens)
        #        while curr_len < lt:
        #            if j >= len(tokens):
        #                print(j, t, tokens, orig_tokens)
        #            T = tokens[j]
        #            if '##' in T:
        #                T = T[2 : ]
        #            if T == '[UNK]':
        #                T = t[len(curr_token)]
        #            curr_len += len(T)
        #            curr_token += T
        #            j += 1
        #        assert(curr_token == t), (sent, tokens, total_sent_length, curr_token, t)
        for k in range(old_j, j):
            mask_ids[k + 1] = old_j + 1
    return mask_ids


def get_n_subwords(sent, tokenizer):
    tokens = tokenizer.tokenize(sent)
    n_tokens = len(tokens)
    mask_ids = make_mask(sent, tokens, n_tokens + 1, tokenizer)[1:]
    n_subwords = []
    for i in range(n_tokens):
        if (i == 0) or (mask_ids[i] != mask_ids[i - 1]):
            n_subwords.append(0)
        n_subwords[-1] += 1
    return n_subwords
