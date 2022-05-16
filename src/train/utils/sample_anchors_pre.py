def sample_anchors_pre(df, n_samples= 256, neg_ratio= 0.5):
    
    '''
    
    Sample total of n samples across both BG and FG classes.
    If one of the classes have less samples than n/2, we will sample from majority class to make up for short.

    Args:
    df with column named labels_anchors, containing 1 for foreground and 0 for background
    n_samples: number of samples to take in total. default 256, so 128 BG and 128 FG.
    neg_ratio: 1/2
    
    '''

    n_fg = int((1-neg_ratio) * n_samples)
    n_bg = int(neg_ratio * n_samples)
    fg_list = [x for x in df['labels_anchors'] if x == 1]
    bg_list = [x for x in df['labels_anchors'] if x == 0]

    # check if we have excessive positive samples
    if len(fg_list) > n_fg:
        # mark excessive samples as -1 (ignore)
        ignore_index = fg_list[n_bg:]
        df.loc[ignore_index, "labels_anchors"] = -1

    # sample background examples if we don't have enough positive examples to match the anchor batch size
    if len(fg_list) < n_fg:
        diff = n_fg - len(fg_list)
        # add remaining to background examples
        n_bg += diff

    # check if we have excessive background samples
    if len(bg_list) > n_bg:
        # mark excessive samples as -1 (ignore)
        ignore_index = fg_list[n_bg:]
        df.loc[ignore_index, "labels_anchors"] = -1