"""
@ MedicalIP Inc
=== Developed by Monib Sediqi :
    04-02-2022
    1. Added 3D preprocessing
    2. Ensure multiples of 16 - utils
    3. Volume resize - utils
    4. Fast Affine Alignment algorithm
"""

import numpy as np
import glob, os
import pydicom
import scipy
from ants import registration, from_numpy, pad_image


def read_dicom_files(dicom_dir):
    try:
        dicom_files = glob.glob(os.path.join(dicom_dir, "*.*"))
        sorted_dicom_files = sorted(dicom_files)
        stacked_dicom = [pydicom.dcmread(dicom_file) for dicom_file in sorted_dicom_files]
        return stacked_dicom
    except IndexError as e:
        print(f"{e}, at path {dicom_dir}")
        return None

def resize_data_volume(data, dim_list):
    """"
    Resize the data to the dim size
    Note: the ratios are not maintained because the voxels have different size and the spatial dimensions are also different
    """
    # data = np.moveaxis(data, -1, 0)
    # print(f'moved axis data shape: {data.shape}')
    height, width, depth  = data.shape
    scale = [dim_list[0] * 1.0/height, dim_list[1] * 1.0 / width, dim_list[2] * 1.0/depth]
    return scipy.ndimage.interpolation.zoom(data, scale, order = 0)

def fast_affine(image_A, image_B, verbose=False, reg_type="AffineFast"):
    """
    Affine alignment - rigid (rotation and translation) + scale - Global alignment of a moving image to the domain of a fixed image

    """
    flag = 0

    image_A = from_numpy(image_A)
    image_B = from_numpy(image_B)

    if image_A.shape[2] > image_B.shape[2]:

        image_A = pad_image(image_A, pad_width=(60, 60, 60), value=0.0)
        moved_dict = registration(moving= image_A, fixed = image_B, reg_type="AffineFast")
        moved = moved_dict['warpedmovout']
        return moved[...], image_B[...], flag
    else:

        image_B = pad_image(image_B, pad_width=(60, 60, 60), value=0.0)
        moved_dict = registration(moving = image_B, fixed = image_A, reg_type="AffineFast")
        moved = moved_dict['warpedmovout']
        flag = 1
        return moved[...], image_A[...], flag

def crop_non_zero_3d(image_3d):
    """
    crop non_zero area of (for now) 3d image. Later ndimage
    params:
    image_3d: 3d image in the form of [h, w, z]

    return:
    """
    indices_3d = np.nonzero(image_3d)
    xyz_min = np.min(indices_3d, 1)
    xyz_max = np.max(indices_3d, 1)

    cropped_3d = image_3d[xyz_min[0]: xyz_max[0], xyz_min[1]:xyz_max[1], xyz_min[2]:xyz_max[2]]
    return cropped_3d

def match_spatial_size(image_3d_A, image_3d_B):
    # -------------------- Make sure the spatial size of images matches each other --------------------

    if image_3d_A.shape[0] < image_3d_B.shape[0]:
        image_3d_A = np.pad(image_3d_A, ((0, abs(image_3d_B.shape[0] - image_3d_A.shape[0])),
                                         (0, 0), (0, 0)), 'constant', constant_values=(-1000, -1000))

    if image_3d_A.shape[0] > image_3d_B.shape[0]:
        image_3d_B = np.pad(image_3d_B, ((0, abs(image_3d_B.shape[0] - image_3d_A.shape[0])),
                                         (0, 0), (0, 0)), 'constant', constant_values=(-1000, -1000))

    if image_3d_A.shape[1] < image_3d_B.shape[1]:
        image_3d_A = np.pad(image_3d_A, ((0, 0),
                                         (0, abs(image_3d_B.shape[1] - image_3d_A.shape[1])), (0, 0)), 'constant', constant_values=(-1000, -1000))

    if image_3d_A.shape[1] > image_3d_B.shape[1]:
        image_3d_B = np.pad(image_3d_B, ((0, 0),
                                         (0, abs(image_3d_B.shape[1] - image_3d_A.shape[1])), (0, 0)), 'constant', constant_values=(-1000, -1000))

    if image_3d_A.shape[2] < image_3d_B.shape[2]:
        image_3d_A = np.pad(image_3d_A, ((0, 0),
                                         (0, 0), (0, abs(image_3d_B.shape[2] - image_3d_A.shape[2]))), 'constant', constant_values=(-1000, -1000))

    if image_3d_A.shape[2] > image_3d_B.shape[2]:
        image_3d_B = np.pad(image_3d_B, ((0, 0),
                                         (0, 0), (0, abs(image_3d_B.shape[2] - image_3d_A.shape[2]))), 'constant', constant_values=(-1000, -1000))

    return image_3d_A, image_3d_B

def ensure_multiples_of_16(image_3d):
    """
    ### NOTE: Adds path on either side of the data
    Ensures that the input image shape is multiple of 16
    image_3d: numpy 3d array
    """
    x,y,z = image_3d.shape
    if x > 512 or y > 512 or z > 512:
        print("Supports image size of up to 512")
    m_16 = list(range(16, 528, 16))
    img_cropped_3d = image_3d

    new_dimension = list(img_cropped_3d.shape)

    for d in range(len(new_dimension)):
        if img_cropped_3d.shape[d] not in m_16:
            for m in m_16:
                if m > img_cropped_3d.shape[d]:
                    new_dimension[d] = m
                    break
    padding = np.array(new_dimension) - np.array(img_cropped_3d.shape)

    new_pad = []
    for pad in padding:
        if pad % 2 == 0:  # the number is even
            p1 = pad // 2
            p2 = pad // 2
            new_pad.append((p1, p2))
        else:  # the number is odd
            p1 = int((pad - 1) / 2)  # (5 - 1) / 2 = 2
            p2 = int(((pad - 1) / 2) + 1)  # (5 - 1)/ 2 => 2 + 1 = 3
            new_pad.append((p1, p2))

    padded_3d = np.pad(img_cropped_3d, new_pad, 'constant', constant_values=(-1000, -1000))
    print('padded 3d shape', padded_3d.shape)
    return padded_3d

def ensure_multiples_of_16_v2(image_3d, is_mask = False):
    """
    ### Note: Adds pad on one side of the data
    Ensures that the input image shape is multiple of 16
    image_3d: numpy 3d array
    """

    x, y, z = image_3d.shape
    if x > 512 or y > 512 or z > 512:
        print("Supports image size of up to 512")
    m_16 = list(range(16, 528, 16))
    img_cropped_3d = image_3d

    new_dimension = list(img_cropped_3d.shape)

    for d in range(len(new_dimension)):
        if img_cropped_3d.shape[d] not in m_16:
            for m in m_16:
                if m > img_cropped_3d.shape[d]:
                    new_dimension[d] = m
                    break
    padding = np.array(new_dimension) - np.array(img_cropped_3d.shape)

    new_pad = []
    for pad in padding:
        if pad % 2 == 0:  # the number is even
            p1 = 0
            p2 = pad
            new_pad.append((p1, p2))
        else:  # the number is odd
            p1 = 0
            p2 = pad  # (5 - 1)/ 2 => 2 + 1 = 3
            new_pad.append((p1, p2))
    if is_mask:
        padded_3d = np.pad(img_cropped_3d, new_pad, 'constant', constant_values=(0, 0))

    else:
        padded_3d = np.pad(img_cropped_3d, new_pad, 'constant', constant_values=(-1000, -1000))
    return padded_3d

# ----------------- To check later -------------------------
def pad(array, shape):
    """
    Zero-pads an array to a given shape. Returns the padded array and crop slices.
    """
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape)])
    padded[slices] = array

    return padded, slices


def resize(array, factor, batch_axis=False):
    """
    Resizes an array by a given factor. This expects the input array to include a feature dimension.
    Use batch_axis=True to avoid resizing the first (batch) dimension.
    """
    if factor == 1:
        return array
    else:
        if not batch_axis:
            dim_factors = [factor for _ in array.shape[:-1]] + [1]
        else:
            dim_factors = [1] + [factor for _ in array.shape[1:-1]] + [1]
        return scipy.ndimage.interpolation.zoom(array, dim_factors, order=0)


def dice(array1, array2, labels=None, include_zero=False):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem

def affine_shift_to_matrix(trf, resize=None, unshift_shape=None):
    """
    Converts an affine shift to a matrix (over the identity).
    To convert back from center-shifted transform, provide image shape
    to unshift_shape.
    TODO: make ND compatible - currently just 3D
    """
    matrix = np.concatenate([trf.reshape((3, 4)), np.zeros((1, 4))], 0) + np.eye(4)
    if resize is not None:
        matrix[:3, -1] *= resize
    if unshift_shape is not None:
        T = np.zeros((4, 4))
        T[:3, 3] = (np.array(unshift_shape) - 1) / 2
        matrix = (np.eye(4) + T) @ matrix @ (np.eye(4) - T)
    return matrix



def dist_trf(bwvol):
    """
    Computes positive distance transform from positive entries in a logical image.
    """
    revbwvol = np.logical_not(bwvol)
    return scipy.ndimage.morphology.distance_transform_edt(revbwvol)

def signed_dist_trf(bwvol):
    """
    Computes the signed distance transform from the surface between the binary
    elements of an image
    NOTE: The distance transform on either side of the surface will be +/- 1,
    so there are no voxels for which the distance should be 0.
    NOTE: Currently the function uses bwdist twice. If there is a quick way to
    compute the surface, bwdist could be used only once.
    """

    # get the positive transform (outside the positive island)
    posdst = dist_trf(bwvol)

    # get the negative transform (distance inside the island)
    notbwvol = np.logical_not(bwvol)
    negdst = dist_trf(notbwvol)

    # combine the positive and negative map
    return posdst * notbwvol - negdst * bwvol


def vol_to_sdt(X_label, sdt=True, sdt_vol_resize=1):
    """
    Computes the signed distance transform from a volume.
    """

    X_dt = signed_dist_trf(X_label)

    if not (sdt_vol_resize == 1):
        if not isinstance(sdt_vol_resize, (list, tuple)):
            sdt_vol_resize = [sdt_vol_resize] * X_dt.ndim
        if any([f != 1 for f in sdt_vol_resize]):
            X_dt = scipy.ndimage.interpolation.zoom(X_dt, sdt_vol_resize, order=1, mode='reflect')

    if not sdt:
        X_dt = np.abs(X_dt)

    return X_dt

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




# ------------------------ OTHER USEFUL METHODS ------------------------

def volcrop(vol, new_vol_shape=None, start=None, end=None, crop=None):
    """
    crop a nd volume.
    Parameters
    ----------
    vol : nd array
        the nd-dimentional volume to crop. If only specified parameters, is returned intact
    new_vol_shape : nd vector, optional
        the new size of the cropped volume
    crop : nd tuple, optional
        either tuple of integers or tuple of tuples.
        If tuple of integers, will crop that amount from both sides.
        if tuple of tuples, expect each inner tuple to specify (crop from start, crop from end)
    start : int, optional
        start of cropped volume
    end : int, optional
        end of cropped volume
    Returns
    ------
    cropped_vol : nd array
    """

    vol_shape = np.asarray(vol.shape)

    # check which parameters are passed
    passed_new_vol_shape = new_vol_shape is not None
    passed_start = start is not None
    passed_end = end is not None
    passed_crop = crop is not None

    # from whatever is passed, we want to obtain start and end.
    if passed_start and passed_end:
        assert not (passed_new_vol_shape or passed_crop), \
            "If passing start and end, don't pass anything else"

    elif passed_new_vol_shape:
        # compute new volume size and crop_size
        assert not passed_crop, "Cannot use both new volume size and crop info"

        # compute start and end
        if passed_start:
            assert not passed_end, \
                "When giving passed_new_vol_shape, cannot pass both start and end"
            end = start + new_vol_shape

        elif passed_end:
            assert not passed_start, \
                "When giving passed_new_vol_shape, cannot pass both start and end"
            start = end - new_vol_shape

        else:  # none of crop_size, crop, start or end are passed
            mid = np.asarray(vol_shape) // 2
            start = mid - (np.asarray(new_vol_shape) // 2)
            end = start + new_vol_shape

    elif passed_crop:
        assert not (passed_start or passed_end or new_vol_shape), \
            "Cannot pass both passed_crop and start or end or new_vol_shape"

        if isinstance(crop[0], (list, tuple)):
            end = vol_shape - [val[1] for val in crop]
            start = [val[0] for val in crop]
        else:
            end = vol_shape - crop
            start = crop

    elif passed_start:  # nothing else is passed
        end = vol_shape

    else:
        assert passed_end
        start = vol_shape * 0

    # get indices. Since we want this to be an nd-volume crop function, we
    # idx = []
    # for i in range(len(end)):
    #     idx.append(slice(start[i], end[i]))

    # special case 1, 2, 3 since it's faster with slicing
    if len(start) == 1:
        rvol = vol[start[0]:end[0]]
    elif len(start) == 2:
        rvol = vol[start[0]:end[0], start[1]:end[1]]
    elif len(start) == 3:
        rvol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    elif len(start) == 4:
        rvol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2], start[3]:end[3]]
    elif len(start) == 5:
        rvol = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2],
                   start[3]:end[3], start[4]:end[4]]
    else:
        idx = range(start, end)
        rvol = vol[np.ix_(*idx)]

    return rvol

