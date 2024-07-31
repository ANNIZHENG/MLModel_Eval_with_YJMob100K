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

1. Sign into NYU HPC: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc

2. Clone this repository

```
git clone https://github.com/ANNIZHENG/MLModel_Eval_with_YJMob100K.git
```

3. Request resourses (1 hour, 4 cores, 4 GPU)

```
srun -t 1:00:00 -c 4 --mem=16000 --gres=gpu:4 --pty /bin/bash
```

4. Load CUDA and Python3 packages

```
module spider cuda  # Check cuda availability

module load cuda/11.6.2
```

```
module spider python  # Check python availability

module load python/intel/3.8.6
```

5. Set-up and Activate Virtual Environment: 

```
virtualenv --system-site-packages -p python3 ./venv  # Create Virtual Environment (OPTIONAL)

source ./venv/bin/activate
```

6. Install PyTorch

```
pip install torch  # Install PyTorch (OPTIONAL)
```

## Run Files (Model Training and Testing)

```
cd MLModel_Eval_with_YJMob100K
python CustomTransformer_with_YJMob100K.py # train and test the Transformer
python LSTM_with_YJMob100K.py              # train and test the LSTM
python Baseline.py                         # test the Baseline model
```