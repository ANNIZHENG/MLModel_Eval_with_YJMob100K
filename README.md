# ML Model Evaluation with YJMob100K

## Organization

```
|-- Baseline.py                         # Baseline model (haven't fixed the train-test split yet)
|-- CustomTransformer_with_YJMob100K.py # Created a custom Transformer for training YJMob100K data
|-- DataAnalysis_POI.ipynb              # Jupyter Notebook file for POI data analysis
|-- LSTM_with_YJmob100K.py              # Created a built-in LSTM for training YJMob100K data
|-- README.md
```

## Run Baseline

```
python Baseline.py
```

## Environment Set-Up (for Running LSTM and Transformer)

### 1. Sign into NYU HPC: 

https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc

### 2. Clone Repository

```
git clone https://github.com/ANNIZHENG/MLModel_Eval_with_YJMob100K.git
```

### 3. Request resourses (2 hour, 4 cores, 1 GPU)

```
srun -t 2:00:00 -c 4 --mem=16000 --gres=gpu:1 --pty /bin/bash
```

### 4. Check CUDA and Python Availability (Optional)

```
module spider cuda
module spider python
```

### 5. Load necessary packages

```
module load cuda/11.6.2
module load python/intel/3.8.6
```

### 6. Set up Virtual Environment (Optional)

```
virtualenv --system-site-packages -p python3 ./venv
pip install torch
```

### 7. Activate Virtual Environment

```
source ./venv/bin/activate
```

### 8. Run Files

```
python <FILE_NAME>.py
```