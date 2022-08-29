# Multi-phase CT scan registration
A UNet based CT scan (Multiphase CT scan and lung CT) registration using Pytorch and Tensorflow

# Network 1 Architecture 
<p align='center'>
  <img src="https://user-images.githubusercontent.com/42628945/185793510-ed4360f9-77f7-4818-9f82-5596985bcb54.png" width=1000)
</p>  

# Network 2 Architecture 


## Requirements
- pytorch
- Tensorflow >= 2.2
- numpy 
- pydicom
- SimpleITK
- pillow

## Visual results - Delayed2Portal
| Moving (M)                                                                                                       |                                                      Fixed (F)                                                      |                                                       Moved (MD)                                                        |                                                    Diff(M - F)                                                    |                                                   Diff(MD - F)                                                    |Flow map|Quiver Map|
|:-------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---:|:---:|
|![01_d](https://user-images.githubusercontent.com/42628945/185793840-a02fa614-0e0b-49a3-ba4f-c384c4c5dcab.jpg)|![01_p(f)](https://user-images.githubusercontent.com/42628945/185793841-7bee7ec1-6326-4a48-8d2d-ffc9aa75e946.jpg)|![01_d(mvd)](https://user-images.githubusercontent.com/42628945/185793835-ebc0b317-836e-40ef-b694-556ba3749faa.jpg) |![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg) |![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg)|![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg)|
|![mv01](https://user-images.githubusercontent.com/42628945/187100104-56d0b378-e4bb-42af-b4d6-d293afccfdc9.jpeg)|![fx01](https://user-images.githubusercontent.com/42628945/187100107-f50b9ab3-c536-4851-9fed-386d18f62f85.jpeg)|![md01](https://user-images.githubusercontent.com/42628945/187100122-db202ad2-515a-4e77-a351-d56c5c9ff59c.jpeg)|||![qv1](https://user-images.githubusercontent.com/42628945/187100131-39f9d49b-3666-4ba6-8fc6-e0326c5f3ee3.png)|![fl01](https://user-images.githubusercontent.com/42628945/187100135-341ec78e-6041-4c68-aa5c-029f48eea77a.png)
|![mv02](https://user-images.githubusercontent.com/42628945/187100238-3d13726b-6cb4-4292-9c45-b8e0a15bea72.jpeg)|![fx02](https://user-images.githubusercontent.com/42628945/187100244-0e5ca6ee-031e-4cc9-ab28-6830a96ff08f.jpeg)|![md02](https://user-images.githubusercontent.com/42628945/187100265-612e8bb2-feab-43f1-be31-f2872bc7f778.jpeg)|||![qv02](https://user-images.githubusercontent.com/42628945/187100250-be81128f-9fd7-4e77-be25-f9ebc90954c3.png)|![fl02](https://user-images.githubusercontent.com/42628945/187100252-9c23d214-916b-491f-8cdf-0ea791896493.png)
|![mv03](https://user-images.githubusercontent.com/42628945/187100365-c0886fd8-7280-4041-b818-0f0b135fffe9.jpeg)|![fx03](https://user-images.githubusercontent.com/42628945/187100368-5a3a2a95-b990-4a5b-82f3-4744c1aef341.jpeg)|![md03](https://user-images.githubusercontent.com/42628945/187100372-2c0366a7-b8b8-4ed0-8d92-988362fbb140.jpeg)|||![qv03](https://user-images.githubusercontent.com/42628945/187100377-03e3053e-b3d4-4ced-85f1-5f518432048c.png)|![fl03](https://user-images.githubusercontent.com/42628945/187100380-63776605-9779-4c1f-b742-cfdcb2229c97.png)

## Implementations
- Random split of data into Train & Val sets 
- Data preprocessing methods: Slices reversion & mask generation  
- U-Net for CNN model  
- Train the model from scratch on Train set and validate on validation set 
- Best model is saved based on the MAE evaluation in validation data
- Tensorboard visualization

## Documentation
### Dataset

### Directory Hierarchy
``` 
.
├── srcImageReg
|   ├── lung_reg
|   |   
|   ├── tf
|   |   ├── cym
|   |   ├── vxm
|   ├── torch
|   |   ├── cym
|   |   ├── vxm
|   |   |   ├── data_preparation
|   |   |   |   ├── creat_data_3d.py
|   |   |   |   ├── load_3d.py
|   |   |   |   ├── preprocess_3d.py
|   |   |   |   ├── utils.py
|   |   |   ├── logs
|   |   |   ├── others
|   |   |   ├── pytorch
|   |   |   |   ├── build_model.py
|   |   |   |   ├── losses.py
|   |   |   |   ├── metrics.py
|   |   |   |   ├── model.py
|   |   |   ├── scripts
|   |   |   |   ├── infer.sh
|   |   |   |   ├── runme.sh
|   |   |   ├── args.py
|   |   |   ├── ndutils.py
|   |   |   ├── train_val.py
```  

### Data Preprocessing
Use `data_pre.ipynb` to create and prepare data. Example usage:
```
run data_pre.ipynb
```

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449965-b583d380-635b-11e9-97c1-fc3e691cae2e.png" width=800)
</p> 
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449970-c16f9580-635b-11e9-9737-0ab8326e4b40.png" width=800)
</p> 
  
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56449971-c896a380-635b-11e9-8657-195451fb7336.png" width=800)
</p> 
  
- Binary head mask using [Otsu auto-thresholding](https://pdfs.semanticscholar.org/fa29/610048ae3f0ec13810979d0f27ad6971bdbf.pdf) 

### Training Network
Use `runme.sh` to train the model. Example usage:
```
bash runme.sh 
```

- CUDA_VISIBLE_DEVICES=0,1,2,3 python train_val.py' 
- `--train_A_data_path `: path to MRI train set
- `--train_B_data_path`: path to CT train set 
- `val_A_data_path`: path to MRI val set
- `val_B_data_path`: path to CT val set
- `val_data_save_path`: result
- `exp-dir`: path to save checkpoints
- `exp_name`:  experiment name
- `data-parallel`: use for multiple gpu training (use only when multiple GPUs are available) 
- `norm_method`: normalization method (default min-max)
- `lr`: learning rate
- `num_epochs_net`: number of epochs 
- `batch-size`: define batch size
- `switch_residualpath`: whether to use residual path or not (1 to use and 0 to not use)
- `debug`: set the verbosity of the debug (1 to turn on debugging and 0 to turn it off)  


### Tensorboard Visualization
Evaluation of validation data during training process.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56470728-8cab2d80-6484-11e9-8e61-b46c11a6942d.png" width=1000)
</p>  

Total loss, data loss and regularization term in each iteration.  

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/56470734-96cd2c00-6484-11e9-92c4-e7166a83838a.png" width=1000)
</p>  
  

### Citation
```
  @misc{Monib Sediqi MedicalImageRegistration,
    author = {Monib Sediqi},
    title = {Medical Image Registration },
    year = {2022},
    url = {https://github.com/Monibsediqi/MRI2CT_synthesis_pytorch},
  }
```

## License
Copyright (c) 2022 Monib Sediqi. Contact me (monib.korea@gmail.com) for commercial use (or rather any use that is not academic research). Free for academic research use only, as long as proper attribution is given and this copyright notice is retained.
