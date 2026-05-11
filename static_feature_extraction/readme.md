## Part 1 the dataset structure

#### 带注释版

```text
├── 1_datasets                # Main dataset directory containing multiple sub-datasets. Each sub-dataset consists of image files with consecutive numerical names
│   ├── dataset1
│   │   ├── 2.png
│   │   ├── 3.png
│   │   └── ...
│   ├── dataset2
│   ├── dataset3
│   └── ...
│
├── 2_seg                     # Directory containing segmented images after processing. Each sub-dataset folder contains segmented results organized by original image names
│   ├── dataset1
│   │   ├── 2                 # Segmented results for image "2.png"
│   │   │   ├── 0_3380_2382_3566_2560_26140.tif   # Naming format: {idx}_{xmin}_{ymin}_{xmax}_{ymax}_{mask_area}.tif
│   │   │   ├── 1_1853_2389_2053_2586_30240.tif
│   │   │   ├── 2_2807_2708_2988_2890_25849.tif
│   │   │   └── ...
│   │   ├── 3                 # Segmented results for image "3.png"
│   │   └── ...
│   ├── dataset2
│   ├── dataset3
│   └── ...
│
├── 3_cla                     # Directory containing classified cell images grouped by cell identity. Each cell is tracked across consecutive frames
│   ├── dataset1
│   │   ├── 0                 # Cell with ID 0 tracked across multiple frames
│   │   │   ├── 0_2_2807_2708_2988_2890_25849.tif   # Naming format: {cell_id}_{first_frame}_{xmin}_{ymin}_{xmax}_{ymax}_{mask_area}.tif
│   │   │   ├── 0_3_2808_2708_2989_2889_25889.tif
│   │   │   ├── 0_4_2807_2708_2989_2890_25929.tif
│   │   │   └── ...
│   │   ├── 2                 # Cell with ID 2 tracked across multiple frames
│   │   └── ...
│   ├── dataset2
│   ├── dataset3
│   └── ...
│
├── 4_background              # Directory containing cell images with added background and mask layers, prepared for MATLAB processing
│   ├── dataset1
│   │   ├── 0                 # Cell with ID 0 with background and masks
│   │   │   ├── 1.tif         # Sequentially numbered images for MATLAB compatibility
│   │   │   ├── 2.tif
│   │   │   ├── 3.tif
│   │   │   ├── ...
│   │   │   ├── mask.tif      # Mask layer 1
│   │   │   ├── mask2.tif     # Mask layer 2
│   │   │   ├── mask3.tif     # Mask layer 3
│   │   │   └── background.tif # Background layer
│   │   ├── 4                 # Cell with ID 4 with background and masks
│   │   └── ...
│   ├── dataset2
│   ├── dataset3
│   └── ...
│
├── 1_SAM.py                  # Script for image segmentation using SAM
├── 2_cla1.py                 # Script for cell classification and tracking
├── 3_add_background.py       # Script for adding background to cell images
├── 4_mask.py                 # Script for generating mask layers
├── 5_for_matlab.py           # Script for preparing data for MATLAB pipeline
└── background.png            # Background image template