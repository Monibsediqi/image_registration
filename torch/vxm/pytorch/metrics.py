import numpy as np


def dice(vol1, vol2, labels = None, nargout = 1):
    """
    Dice [1] volume overlap metric
    The default is to *not* return a measure for the background layer (label = 0)
    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ---------------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)

    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels

    nargout : optional control of exp_output arguments. if 1, exp_output Dice measure(s).
        if 2, exp_output tuple of (Dice, labels)

    method : 'iter' (iterating through channels) or 'hist2' (using a histogram)

    Return
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    """

    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    print('len of labels', len(labels))

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)



# -------------------------- Jacobian determinant ---------------------------
def ndgrid(*args, **kwargs):
    """
    Disclaimer: This code is taken directly from the scitools package [1]
    Since at the time of writing scitools predominantly requires python 2.7 while we work with 3.5+
    To avoid issues, we copy the quick code here.
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args, **kwargs)

def volsize2ndgrid(volsize):
    """
    return the dense nd-grid for the volume with size volsize
    essentially return the ndgrid fpr
    """
    ranges = [np.arange(e) for e in volsize]
    return ndgrid(*ranges)

def jacobian_determinant(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
##############################################################
#                        SANITY CHECK                        #
##############################################################

if __name__ == "__main__":
    path_vol1 = '../sample_data/train_data_A.npz'
    path_vol2 = '../sample_data/train_data_B.npz'

    data_1 = np.load(path_vol1)
    data_2 = np.load(path_vol2)

    print(data_1['vol'].shape)
    print(data_2['vol'].shape)

    vol1 = data_1['vol']
    vol2 = data_2['vol']

    device = 'cuda'
    dice_score = dice_torch(vol1, vol2, device = device)
    print(dice_score.shape)