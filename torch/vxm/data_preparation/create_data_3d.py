"""
initiated by JM Kim, Ph.D., MedicalIP, Inc.
===initial: 27-August-2021
--------------------------------------------------------------------------
=== Modified by Monib Sediqi : 04-02-2022
    1. Added 3D preprocessing
    2. Ensure multiples of 16 - utils
    3. Volume resize - utils
    4. Fast Affine Alignment algorithm
    5. Add mask for dice score calculation
"""

from data_preparation.load_3d import Image_3D_Data
from torch.utils.data import DataLoader
from data_preparation.preprocess_3d import Preprocess_3D


def create_datasets_3d(args):

    train_data = Image_3D_Data(
        root_A=args.train_moving_data,
        root_B=args.train_fixed_data,
        m_root_A=args.train_moving_mask,
        m_root_B=args.train_fixed_mask,
        preprocess_3d=Preprocess_3D(method=args.norm_method, scale_factor = args.sf),
        data_type= args.data_type,
    )
    print('train_data')
    val_data = Image_3D_Data(
        root_A=args.val_moving_data,
        root_B=args.val_fixed_data,
        m_root_A=args.val_moving_mask,
        m_root_B=args.val_fixed_mask,
        preprocess_3d=Preprocess_3D(method=args.norm_method, scale_factor = args.sf),
        data_type=args.data_type,
    )
    print('val_data')

    return train_data, val_data


def create_data_loaders_3d(args):

    train_data, val_data = create_datasets_3d(args)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn= None,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    display_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )

    print(
        f"created data loaders (number of 3D scans), train loader: {len(train_loader.dataset)}, val loader: {len(val_loader.dataset)}, display loader :  {len(display_loader.dataset)},")
    return train_loader, val_loader, display_loader

def create_infer_datasets_3d(args):

    print('train_data')
    infer_data = Image_3D_Data(
        root_A=args.val_moving_data,
        root_B=args.val_fixed_data,
        m_root_A=args.val_moving_mask,
        m_root_B=args.val_fixed_mask,
        preprocess_3d=Preprocess_3D(method=args.norm_method, scale_factor = args.sf),
        data_type=args.data_type,
    )
    print('val(infer)_data')

    return infer_data

def create_infer_data_loaders_3d(args):
    val_data = create_infer_datasets_3d(args)

    infer_loader = DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )

    print(f"createD data loaders (number of 3D scans) infer loader :  {len(infer_loader.dataset)},")
    return  infer_loader

