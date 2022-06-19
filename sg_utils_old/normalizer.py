import numpy as np
def get_norm_stat_for_frame_repr_list(repr_list, feat_dim):
    """
    mel_spec: (D, T)
    """
    sum_vec = np.zeros(feat_dim)
    sqsum_vec = np.zeros(feat_dim)
    count = 0
    for feat_mat in repr_list:
        assert(feat_mat.shape[0]) == feat_dim
        
        feat_sum = np.sum(feat_mat, axis=1)
        feat_sqsum = np.sum(feat_mat**2, axis=1)

        sum_vec += feat_sum
        sqsum_vec += feat_sqsum

        count += feat_mat.shape[1]

    feat_mean = sum_vec / count
    feat_var = (sqsum_vec / count) - (feat_mean**2)

    return feat_mean, feat_var

def get_norm_stat_for_melspec(spec_list, feat_dim=128):
    feat_mean, feat_var = get_norm_stat_for_frame_repr_list(spec_list, feat_dim)
    return feat_mean, feat_var