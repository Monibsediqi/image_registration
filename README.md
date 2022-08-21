# Multi-phase CT scan registration
A UNet based CT scan (Multiphase CT scan and lung CT) registration using Pytorch and Tensorflow

# Network 1 Architecture 
<p align='center'>
  <img src="https://user-images.githubusercontent.com/42628945/184533461-9663ea09-4de6-4826-95c0-efb834b9a273.jpg" width=1000)
</p>  

# Network 2 Architecture 
<p align='center'>
  <img src="https://user-images.githubusercontent.com/42628945/184533461-9663ea09-4de6-4826-95c0-efb834b9a273.jpg" width=1000)
</p>  

## Requirements
- pytorch
- Tensorflow >= 2.2
- numpy 
- pydicom
- SimpleITK
- pillow

## Visual results
|MRI|CBCT | Synthetic CT | 
|:---:|:---:|:---:|
| ![IMG0070](https://user-images.githubusercontent.com/42628945/184530412-e2ffbd67-ebec-4ee3-8cd4-0194ba8cf8a3.jpg)|![Cleaned_0000070](https://user-images.githubusercontent.com/42628945/184530432-ba3aa975-0050-49b5-b722-1c722619675f.jpg)| ![Cleaned_0000067](https://user-images.githubusercontent.com/42628945/184530458-ddbe3798-d016-42b8-bd3f-7d0f9f38d718.jpg)|
|![IMG0086](https://user-images.githubusercontent.com/42628945/184530674-40ea7848-7a36-4756-9224-d75d9589921d.jpg)|![Cleaned_0000086](https://user-images.githubusercontent.com/42628945/184530682-ad338522-82be-4236-8064-6f537fba057b.jpg)|![Cleaned_0000068](https://user-images.githubusercontent.com/42628945/184530687-f9632fc6-6c31-4fd5-bd47-20107dc459f7.jpg)|
|![IMG0111](https://user-images.githubusercontent.com/42628945/184531047-b66547aa-3d79-42cb-850d-b59108e6b896.jpg)|![Cleaned_0000111](https://user-images.githubusercontent.com/42628945/184531059-59d32fb9-be44-4f2a-95ea-0d2bdcff76cb.jpg)|![Cleaned_00000111](https://user-images.githubusercontent.com/42628945/184531060-69239d86-0ba2-47fd-aedd-055559ba12d4.jpg)
|![IMG0148](https://user-images.githubusercontent.com/42628945/184531127-447417dc-8e63-4447-8e94-04ece84877f6.jpg)|![Cleaned_0000148](https://user-images.githubusercontent.com/42628945/184531131-45b28bdf-93b1-47e0-a68c-c0a703b89266.jpg)|![Cleaned_0000148](https://user-images.githubusercontent.com/42628945/184531189-bfe068c1-acd0-4678-b168-9fb27e099db5.jpg)
|![IMG0204](https://user-images.githubusercontent.com/42628945/184531297-9f3b3082-6738-4fd2-b030-1710425ac165.jpg)|![Cleaned_0000204](https://user-images.githubusercontent.com/42628945/184531302-e80d8ea4-5338-4788-8e3b-623c2d81e8b5.jpg)|![Cleaned_0000204](https://user-images.githubusercontent.com/42628945/184531368-e737a048-543c-47fb-8baa-ba15a76302f3.jpg)
|![IMG0259](https://user-images.githubusercontent.com/42628945/184531486-91785a79-dd1b-4902-822b-0d417427264e.jpg)|![Cleaned_0000259](https://user-images.githubusercontent.com/42628945/184531488-c4e487b9-396f-4cb6-81c4-04a85681a876.jpg)|![Cleaned_0000259](https://user-images.githubusercontent.com/42628945/184531490-7e9238a8-c93f-4843-b8d8-f8e0eb4e8abc.jpg)
|![IMG0289](https://user-images.githubusercontent.com/42628945/184531586-5bb5ef90-e002-42d7-a0b8-944122b6cc71.jpg)|![Cleaned_0000289](https://user-images.githubusercontent.com/42628945/184531597-6f12f248-7ce0-492e-bb1a-c6a9ca651790.jpg)|![Cleaned_0000289](https://user-images.githubusercontent.com/42628945/184531602-b636ecbe-81d1-4530-b3f4-a3958e797b78.jpg)
|![IMG0320](https://user-images.githubusercontent.com/42628945/184532316-642b6179-8879-43d5-8a06-638ce0d07642.jpg)|![Cleaned_000320](https://user-images.githubusercontent.com/42628945/184532331-2c47a736-b4c1-4a28-b522-ba3d8d09b1b1.jpg)|![Cleaned_0000320](https://user-images.githubusercontent.com/42628945/184532342-44540464-fdcf-401e-a704-2eb75b0f56ab.jpg)
 

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
├── MRI2CT_synthsis_pytorch
|   ├── dataset
|   |   ├── DataPreprocessing.py
|   |   ├── DataSlicing.py
|   |   ├── create_data.py
|   ├── models
|   |   ├── build_model.py
|   |   ├── unet_model.py
|   ├── script
|   |   ├── data_prep.ipynb
|   |   ├── runme.sh
|   ├── args.py
|   ├── train_val.py

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
