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

### 1. Sign into NYU HPC: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc

### 2. Clone this repository

```
git clone https://github.com/ANNIZHENG/MLModel_Eval_with_YJMob100K.git
```

### 3. Request resourses (2 hour, 4 cores, 1 GPU)

```
srun -t 2:00:00 -c 4 --mem=16000 --gres=gpu:1 --pty /bin/bash
```

### 4. Load CUDA and Python3 packages

Check CUDA availabiilty
```
module spider cuda
```

Load CUDA
```
module load cuda/11.6.2
```

Check Python availability

```
module spider python
```

Load Python

```
module load python/intel/3.8.6
```

### 5. Set Up and Activate Virtual Environment: 

Create Virtual Environment

```
virtualenv --system-site-packages -p python3 ./venv
```

Activate Virtual Environment

```
source ./venv/bin/activate
```

### 6. If not already, install PyTorch

```
pip install torch
```

## Run Files (Model Training and Testing)

```
python CustomTransformer_with_YJMob100K.py 
python LSTM_with_YJMob100K.py
python Baseline.py
```