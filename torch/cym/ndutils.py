import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from data_preparation.utils import read_dicom_files

def resize_data_volume(data, dim_list):
    """"
    Resize the data to the dim size
    Note: the ratios are not maintained because the voxels have different size and the spatial dimensions are also different
    """
    depth, height, width = data.shape
    scale = [dim_list[0] * 1.0/depth, dim_list[1] * 1.0/height, dim_list[2] * 1.0 / width]
    print(scale)
    return scipy.ndimage.interpolation.zoom(data, scale, order = 0)

def bw_grid(vol_shape, spacing, thickness=1):
    """
    draw a black and white ND grid.
    Parameters
    ----------
        vol_shape: expected volume size
        spacing: scalar or list the same size as vol_shape
    Returns
    -------
        grid_vol: a volume the size of vol_shape with white lines on black background
    """

    # check inputs
    if not isinstance(spacing, (list, tuple)):
        spacing = [spacing] * len(vol_shape)
    spacing = [f + 1 for f in spacing]
    assert len(vol_shape) == len(spacing)

    # go through axes
    grid_image = np.zeros(vol_shape)
    for d, v in enumerate(vol_shape):
        rng = [np.arange(0, f) for f in vol_shape]
        for t in range(thickness):
            rng[d] = np.append(np.arange(0 + t, v, spacing[d]), -1)
            grid_image[ndgrid(*rng)] = 1

    return grid_image

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



def meshgridnd_like(in_img, rng_func=range):
    new_shape = list(in_img.shape)
    all_range = [rng_func(i_len) for i_len in new_shape]
    return tuple([x_arr.swapaxes(0, 1) for x_arr in np.meshgrid(*all_range)])

def get_quiver_plot(flow_field, ds_factor = 20):
    """
    Params:
    flow_field: deformation field in the form of np.array: e.g., (512,512,112,3)
    ds_factor = an integer indicating the sparsity of the arrows in the quiver plot
    """
    DS_FACTOR = ds_factor
    flow = np.moveaxis(flow_field, 0, -1)

    c_xx, c_yy, c_zz = [x.flatten()
                        for x in
                        meshgridnd_like(flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, 0])]

    get_flow = lambda i: flow[::DS_FACTOR, ::DS_FACTOR, ::DS_FACTOR, i].flatten()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    ax.quiver(c_xx,
              c_yy,
              c_zz,
              get_flow(0),
              get_flow(1),
              get_flow(2),
              length=0.9,
              normalize=True)
    return fig


def get_2d_quiver(flow_2d, sp_factor = 20):
    """
    flow_2d: Flow filed in 2d+3 format. Example (512,512,3)
    sp_factor = sparsity factor.
    """
    spatial_flow = flow_2d[:, :, 0:2]
    meshg = meshgridnd_like(spatial_flow[::sp_factor, ::sp_factor, 0])
    mesh = np.asarray(meshg)
    mesh_mv = np.moveaxis(mesh, 0, -1)
    meshX = mesh_mv[:, :, 0]
    meshY = mesh_mv[:, :, 1]


    flowX_2d = flow_2d[::sp_factor, ::sp_factor, 0]
    flowY_2d = flow_2d[::sp_factor, ::sp_factor, 1]
    flowZ_2d = flow_2d[::sp_factor, ::sp_factor, 2]

    fig, ax = plt.subplots(figsize=(10,10))

    ax.quiver(meshX, meshY, flowX_2d, flowY_2d, flowZ_2d )

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_aspect('equal')

    return fig


def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax


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


def ensure_multiples_of_16(image_3d):
    """
    Ensures that the input image shape is multiple of 16
    image_3d: numpy 3d array
    """

    multiples_of_16 = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
                       336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624,
                       640, 656, 672, 688, 704, 720, 736, 752, 768, 784]


    img_cropped_3d = crop_non_zero_3d(image_3d)

    new_dimension = [img_cropped_3d.shape[0], img_cropped_3d.shape[1], img_cropped_3d.shape[2]]  # x, y , z

    for d in range(len(new_dimension)):
        if img_cropped_3d.shape[d] not in multiples_of_16:
            for m in multiples_of_16:
                if m > img_cropped_3d.shape[d]:
                    new_dimension[d] = m
                    break
    print(new_dimension)

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

    return np.pad(img_cropped_3d, new_pad, 'constant')

def random_sampler (ndarray, patch_size, num_patches):
    # Generate image patches
    image_patches = []
    locations = []
    a, b, c = ndarray.shape
    for i in range(num_patches):
        a_coor = np.random.randint(0, a - patch_size[0])
        b_coor = np.random.randint(0, b - patch_size[1])
        c_coor = np.random.randint(0, c - patch_size[2])

        patch_image = ndarray[a_coor:a_coor + patch_size[0], b_coor:b_coor + patch_size[1], c_coor:c_coor + patch_size[2]]
        # patch_image[...] = 0
        image_patches.append(patch_image)
        locations.append((a_coor, b_coor, c_coor))
    return image_patches, locations


def uniform_sampler(image, patch_size):
    """
    image: a 3D CT scan image
    patch_size: a tuple of int indicating the patch size. ex (128,128,128
    """
    stride = int(patch_size / 2)

    image_patches = []
    locations = []

    for i in range(0, image.shape[0], stride):
        for j in range(0, image.shape[1], stride):
            for k in range(0, image.shape[2], stride):
                patch_img = image[i:i + patch_size, j:j + patch_size, k:k + patch_size]
                image_patches.append(patch_img)
                locations.append((i, j, k))

    return image_patches, locations

def uniform_sampler_v2(image, patch_size):
    """
    image: a 3D CT scan image
    patch_size: a tuple of int indicating the patch size. ex (128,128,128)
    Note: This patch sampler does not consider z axis
    """
    stride = int(patch_size / 2)

    image_patches = []
    locations = []

    for i in range(0, image.shape[0], stride): # image.shape[0] = 512, i =0
        for j in range(0, image.shape[1], stride): # image.shape[1] = 512
            patch_img = image[i:i + patch_size, j:j + patch_size, ...] #
            image_patches.append(patch_img)
            locations.append((i, j))

    # patch size = 8, stride 4
    # image [0: 0 + 8, 0: 0 + 8, ...], i = 4, image[4:12, 4:12, ...], i=8, image[8:
    #
    #
    #

    return image_patches, locations

def save_flow(flow_np, save_path):
    """
    args:
    flow_np: a 4D flow with type numpy
    """
    for fl in range(flow_np.shape[3]):
        single_flow = flow_np[:, :, :, fl]
        single_flow = np.moveaxis(single_flow, 0, -1)

        single_flow_min = np.min(single_flow)
        single_flow_max = np.max(single_flow)

        single_flow = ((single_flow - single_flow_min) / (single_flow_max - single_flow_min))

        save_flow_path = os.path.join(save_path, f"FL_0000{fl:03}.png")

        plt.imsave(save_flow_path, single_flow, cmap='brg')
        plt.close()


def save2dicom(ref_dicom_files, ndarray, save_path, T_slices = None):
    """
    ref_dicom_files: reference dicom files for meta data
    ndarray: a 3D dimension array to save be saved as dicom files
    save_path = str, path to directory to save dicom files
    T_slices = Total number of slices
    """

    if T_slices is not None:
        slices = T_slices
    else:
        slices = len(ref_dicom_files)

    for i in range(slices):
        dicom_slice = ref_dicom_files[i]
        arr_2d = ndarray[:, :, i]
        # arr_2d[arr_2d < 0] = -1000 # rescale negative values to -1000
        slice_2d = arr_2d.astype(str(dicom_slice.pixel_array.dtype))
        dicom_slice.PixelData = slice_2d.tobytes()
        # dicom_slice.RescaleIntercept = -1024
        dicom_slice.InstanceNumber = i

        save_slices_path = os.path.join(save_path, f"MV_0000{i:03}.dcm")

        dicom_slice.SeriesDescription = "[Research & Science] - Generated Data"
        dicom_slice.save_as(save_slices_path)


#########################################
#           TEST PROGRAM                #
#########################################


if __name__ == "__main__":
    test_ct = "/media/monib/ext1/work2022/Base_Dataset/test/input/val/example_A/HCC_1108_d"
    save_path = "test_ct"
    patch_size = 256
    stacked_dicom_A = read_dicom_files(test_ct)
    image_shape_A = list(stacked_dicom_A[0].pixel_array.shape)
    image_shape_A.append(len(stacked_dicom_A))

    image_3d_A = np.zeros(image_shape_A)
    for j in range(len(stacked_dicom_A)):
        image_3d_A[:, :, j] = stacked_dicom_A[j].pixel_array

    print(f"min: {np.min(image_3d_A)}, max: {np.max(image_3d_A)}")
    print(f"shape: {image_3d_A.shape}")

    image_A_patches, locations_A = uniform_sampler_v2(image_3d_A,
                                                      patch_size=patch_size)

    moved_scan = np.zeros(image_3d_A.shape)
    for loc_idx in range(len(locations_A)):
        a_coor_A, b_coor_A = locations_A[loc_idx]
        moved_scan[a_coor_A:a_coor_A + patch_size, b_coor_A:b_coor_A + patch_size, ...] = image_A_patches[loc_idx]

    save2dicom(stacked_dicom_A, moved_scan, save_path)
    print("saved!")







