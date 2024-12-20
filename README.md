# MMR Project
Efraim Dahl  
Casper Willem Smet  
Maria-Raluca Stanescu  

# Content-Based 3D Retrieval Engine

This repository contains a content-based retrieval engine for 3D models. The engine allows users to query a dataset of 3D models and retrieve results based on content similarity, enabling tasks like object recognition, classification, and similarity search.

## Features
- **3D Model Feature Extraction**: Extracts hand-crafted features from 3D models.
- **Content-Based Search**: Enables optimized similarity-based retrieval of 3D models.
- **User-Friendly GUI**: Search and retrieve models within a GUI.

## Getting Started

### Prerequisites
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html): Ensure you have Conda installed. You can use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for a lightweight installation or [Anaconda](https://www.anaconda.com/products/individual) for the full suite.

### Installation

1. **Clone the Repository**

   Clone the repository (including the shapes git submodule) to your local machine using:
   ```bash
   git clone --recurse-submodules https://github.com/maria-raluca-st/INFOMR-final-project.git
   # git clone https://github.com/maria-raluca-st/INFOMR-final-project.git  # Alternative when you do not want to include the shapes
   cd INFOMR-final-project

   ```

2. **Create the Conda Environment**

   The repository includes an `environment.yml` file with all the dependencies needed to run the engine. To create the environment, use:

   ```bash
   conda env create -f environment.yml
   ```

   This command will:
   - Create a new environment with the name specified in the `environment.yml` file (it will be named `infomr`).
   - Install all necessary dependencies.

3. **Activate the Environment**

   After creating the environment, activate it with:
   ```bash
   conda activate infomr
   ```

4. **Install Additional Packages (Optional)**

   If additional packages are required or the `environment.yml` file is missing some dependencies, you can install them manually:
   ```bash
   conda install <package-name>
   ```
   If no conda repository is available, install them using pip
    ```bash
   pip install <package-name>
   ```
### Adding the Dataset
For the program to run properly it expects the root directory to contain the normalized dataset called normshapes and the non-normalized dataset called shapes. These can be added separately. To normalize in parallel, navigate to the file `mmdbs\feature_extraction.py` in the main function starting in line 333 and uncomment the following section (lines 339-348)
```python
    extract_dataset_features_from_shapes(
        df_manifest,
        shape_directory="../shapes",
        output_file="mesh_features.csv",
        normalize=True,
        output_directory="../normshapes,

    )
    return
``` 
Alternatively, you can normalize (monothreaded) using the `preprocessing_stats.ipynb` notebook and set the variable `normalize_shapes` in the first cell to `True`. This will create a folder called `normashapes` in the root directory of the project. 
### Usage

After setting up the environment and adding the dataset, navigate to the mmdbs folder and start the gui.py file. Please note that the first time executing this will be slow, as it has to build an index of database.

```bash
cd mmdbs
python gui.py
```

The gui has to be used in fullscreen mode. Otherwise part of the control panel will not be visible.
Complete the following steps;
1) Load a model into the GUI 
2) A panel will have appeared where you can control the shapes appearance, remove and normalize it.
3) Press normalize to move the shape into the center of the 3D plot
4) Press Search to search for similar objects. (The object that appears first on the panel list, is the reference object)
5) Press cycle xz, or cycle xy to view all objects on different planes and see information about them.
6) Select the checkboxes above to show the origin axis and a unit cube.

## Environment File Structure

The `environment.yml` file should look something like this:
```yaml
name: infomr
dependencies:
  - python=3.8
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - pytorch
  - torchvision
  - -c pytorch  # Specifies the PyTorch channel for Conda
  - pip:
      - trimesh  # Additional package for 3D mesh processing
```

### Updating the Environment

To add new packages to your environment and update `environment.yml`, use:
```bash
conda env export > environment.yml
```
