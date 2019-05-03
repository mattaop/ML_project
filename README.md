# gangsta_project
### How to install requirements
Create an environment, activate and install requirements:
```bash
conda create -n name_of_environment python=3.6
conda activate name_of_environment
pip install -r requirements.txt
```

### Folder structure
This folder structure required to run the project without any modifications:

```bash
|---- datasets  # Folder for data sets
|     |----- chars74k-lite  # Datasets requiered to train the models
|            |---- a
|            |---- b
|            |...
|            |---- z
|     |----- detection images  # Datasets requiered to do character detection
|---- models
|     |---- CNN.py
|     |---- KNN.py
|     |---- SVM.py
|     |---- Naive_Bayes.py
|---- feature_extraction
|     |---- data_processing.py
|     |---- filters.py
|---- weights
|     |---- character_detection_weights.hdf5
|---- character_classification.py
|---- character_detection.py
|---- plots.py
|---- load_data.py
|---- model_weights.hdf5
```

### Run code
The code should be ran as modules, with the following calls in conda prompt:
```bash
python -m models.CNN -n simple
python -m models.CNN -n residual
python -m models.SVM
python -m character_detection
python -m plots
```
