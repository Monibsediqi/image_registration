# Multi-phase CT scan registration
A UNet based CT scan (Multiphase CT scan and lung CT) registration using Pytorch and Tensorflow

# Schematic View

<p align='center'>
  <img src="https://user-images.githubusercontent.com/42628945/187315498-3bc741a5-24c3-4026-82f8-f4f981092b88.png" width=1000)
</p> 

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

## Visual results
### Delayed to Portal 
| Moving (M)                                                                                                       |                                                      Fixed (F)<br>                                                      |                                                       Moved (MD)                                                        |                                                    Diff(M - F)                                                    |                                                   Diff(MD - F)                                                    |Flow map|Quiver Map|
|:-------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---:|:---:|
|![01_d](https://user-images.githubusercontent.com/42628945/185793840-a02fa614-0e0b-49a3-ba4f-c384c4c5dcab.jpg)|![01_p(f)](https://user-images.githubusercontent.com/42628945/185793841-7bee7ec1-6326-4a48-8d2d-ffc9aa75e946.jpg)|![01_d(mvd)](https://user-images.githubusercontent.com/42628945/185793835-ebc0b317-836e-40ef-b694-556ba3749faa.jpg) |![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg) |![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg)|![01_qvr](https://user-images.githubusercontent.com/42628945/185797116-df0cae70-f3a7-4d7d-be8f-711bedebb36c.jpg) |![01_fl](https://user-images.githubusercontent.com/42628945/185793839-0afa215e-d149-4ee1-a3ca-b4924b1698f7.jpg)|
|![mv01](https://user-images.githubusercontent.com/42628945/187100104-56d0b378-e4bb-42af-b4d6-d293afccfdc9.jpeg)|![fx01](https://user-images.githubusercontent.com/42628945/187100107-f50b9ab3-c536-4851-9fed-386d18f62f85.jpeg)|![md01](https://user-images.githubusercontent.com/42628945/187100122-db202ad2-515a-4e77-a351-d56c5c9ff59c.jpeg)|||![qv1](https://user-images.githubusercontent.com/42628945/187100131-39f9d49b-3666-4ba6-8fc6-e0326c5f3ee3.png)|![fl01](https://user-images.githubusercontent.com/42628945/187100135-341ec78e-6041-4c68-aa5c-029f48eea77a.png)
|![mv02](https://user-images.githubusercontent.com/42628945/187100238-3d13726b-6cb4-4292-9c45-b8e0a15bea72.jpeg)|![fx02](https://user-images.githubusercontent.com/42628945/187100244-0e5ca6ee-031e-4cc9-ab28-6830a96ff08f.jpeg)|![md02](https://user-images.githubusercontent.com/42628945/187100265-612e8bb2-feab-43f1-be31-f2872bc7f778.jpeg)|||![qv02](https://user-images.githubusercontent.com/42628945/187100250-be81128f-9fd7-4e77-be25-f9ebc90954c3.png)|![fl02](https://user-images.githubusercontent.com/42628945/187100252-9c23d214-916b-491f-8cdf-0ea791896493.png)
|![mv03](https://user-images.githubusercontent.com/42628945/187100365-c0886fd8-7280-4041-b818-0f0b135fffe9.jpeg)|![fx03](https://user-images.githubusercontent.com/42628945/187100368-5a3a2a95-b990-4a5b-82f3-4744c1aef341.jpeg)|![md03](https://user-images.githubusercontent.com/42628945/187100372-2c0366a7-b8b8-4ed0-8d92-988362fbb140.jpeg)|||![qv03](https://user-images.githubusercontent.com/42628945/187100377-03e3053e-b3d4-4ced-85f1-5f518432048c.png)|![fl03](https://user-images.githubusercontent.com/42628945/187100380-63776605-9779-4c1f-b742-cfdcb2229c97.png)

### Porta to Arterial 
| Moving (M)                                                                                                       |                                                      Fixed (F)<br>                                                      |                                                       Moved (MD)                                                        |                                                    Diff(M - F)                                                    |                                                   Diff(MD - F)                                                    |Flow map|Quiver Map|
|:-------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---:|:---:|
|![mv01](https://user-images.githubusercontent.com/42628945/187317646-7f8d7da4-29a1-4619-84f8-9c99ed443e2a.jpeg)|![fx01](https://user-images.githubusercontent.com/42628945/187317643-518399c3-88ab-4122-8b58-bd132bcff2d4.jpeg)|![md01](https://user-images.githubusercontent.com/42628945/187317645-b66a931b-794d-44d0-89a2-7a231d24cf55.jpeg)|||![qv01](https://user-images.githubusercontent.com/42628945/187317638-49e38797-61c6-4ee5-a844-02cfecaf7534.jpg)|![fl01](https://user-images.githubusercontent.com/42628945/187317641-9fd4d51a-421d-46f4-b14a-e02982366a2c.png)
|![mv02](https://user-images.githubusercontent.com/42628945/187317964-f0d0107f-d1c7-45e4-af1b-af3887ee6638.jpeg)|![fx02](https://user-images.githubusercontent.com/42628945/187317957-8b373f2f-671b-4315-a193-29d52390c718.jpeg)|![md02](https://user-images.githubusercontent.com/42628945/187317961-fad40e12-5e66-4249-88cb-c99635c99f08.jpeg)|||![qv02](https://user-images.githubusercontent.com/42628945/187317966-fa7627dc-c991-4772-9b91-3c05f10cdbc5.jpg)|![fl02](https://user-images.githubusercontent.com/42628945/187317945-4b73d570-f960-4456-a412-6d097399656a.png)
|![mv03](https://user-images.githubusercontent.com/42628945/187318215-0092e1ce-000a-4049-8508-216a233e1411.jpeg)|![fx03](https://user-images.githubusercontent.com/42628945/187318211-993a5460-7ba6-406b-8581-5fb6e725e1b1.jpeg)|![md03](https://user-images.githubusercontent.com/42628945/187318213-d26c0e3a-be31-4a5e-a691-274ec3194230.jpeg)|||![qv03](https://user-images.githubusercontent.com/42628945/187318217-343e1e5e-c8e7-4c2c-a7e2-7d3b2775ffc5.jpg)|![fl03](https://user-images.githubusercontent.com/42628945/187318206-6ef5abac-0cf5-4823-962d-a86318a6d972.png)
|![mv04](https://user-images.githubusercontent.com/42628945/187318600-9d29bc1a-57e6-4cce-b026-ffc5a1593bac.jpeg)|![fx04](https://user-images.githubusercontent.com/42628945/187318608-5d99e877-6a9d-4891-8904-80e5e04e9ad1.jpeg)|![md04](https://user-images.githubusercontent.com/42628945/187318618-0c75c10e-f4d3-42f6-9d65-1e2ee9fddbd4.jpeg)|||![qv04](https://user-images.githubusercontent.com/42628945/187318626-07338d90-ade7-4f85-ab50-aa2cfbfd8333.jpg)|![fl04](https://user-images.githubusercontent.com/42628945/187318630-5565b9ef-9442-481e-a186-f0f8abdeead9.png)
### Arterial to Delayed
| Moving (M)                                                                                                       |                                                      Fixed (F)<br>                                                      |                                                       Moved (MD)                                                        |                                                    Diff(M - F)                                                    |                                                   Diff(MD - F)                                                    |Flow map|Quiver Map|
|:-------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------:|:---:|:---:|
|![mv0](https://user-images.githubusercontent.com/42628945/189782234-9101a003-66c6-4fe4-8b5c-53153d742f5c.jpg)|![fx0](https://user-images.githubusercontent.com/42628945/189782242-aeebc996-8277-4f2e-ba83-4c37ae3bf740.jpg)|![md0](https://user-images.githubusercontent.com/42628945/189782249-904658db-a0de-41f4-82e9-b4fad0c21ba2.jpg)|||![qv0](https://user-images.githubusercontent.com/42628945/189782615-907f388f-489a-416d-ba46-8c04f5b51004.png)|![fl0](https://user-images.githubusercontent.com/42628945/189782618-fb52d8f4-2c5c-4ba7-8ede-ecb1e06456b4.png)
|![mv1](https://user-images.githubusercontent.com/42628945/189782834-e02c33e1-8102-43ce-99b6-9ae4bf56362d.jpg)|![fx1](https://user-images.githubusercontent.com/42628945/189782849-ae1faad8-fa76-4e28-9a6e-7178eb0bb79f.jpg)|![md1](https://user-images.githubusercontent.com/42628945/189782899-2b34eb90-c69b-405a-bde7-d58cb9586796.jpg)|||![qv1](https://user-images.githubusercontent.com/42628945/189782902-8b009a54-2e46-41de-ab64-7355b88a9094.png)|![fl1](https://user-images.githubusercontent.com/42628945/189782906-b7447b9e-28f0-4459-923f-25d29c562ae1.png)
|![mv2](https://user-images.githubusercontent.com/42628945/189783118-61c73d04-05ac-4c3b-a5fe-c9cea2e6cf04.jpg)|![fx2](https://user-images.githubusercontent.com/42628945/189783132-42b9165b-0cd5-414f-9101-3158f0b18b07.jpg)|![md2](https://user-images.githubusercontent.com/42628945/189783156-e37e6048-7eee-4f47-ad23-ff69371f067c.jpg)|||![qv2](https://user-images.githubusercontent.com/42628945/189783180-4349f03b-cf44-4008-a8d2-863283e207c8.png)|![fl2](https://user-images.githubusercontent.com/42628945/189783189-c4eda58c-18de-40f9-917f-3a4c6332cd29.png)
![mv3](https://user-images.githubusercontent.com/42628945/189785154-fe0951c9-e681-4f5c-b3b2-7d66e2c9e4c7.jpg)|![fx3](https://user-images.githubusercontent.com/42628945/189785161-9a0f6ccf-ad74-4169-b86e-d6aa585dd15f.jpg)|![md3](https://user-images.githubusercontent.com/42628945/189785165-321c0153-01e1-4239-8e37-3e627c931f92.jpg)|||![qv3](https://user-images.githubusercontent.com/42628945/189785170-fdc34f47-24b4-42d6-aa6f-0db4d48fbff7.png)|![fl3](https://user-images.githubusercontent.com/42628945/189785174-107c9f89-b117-418b-9764-36dc4102989a.png)

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
