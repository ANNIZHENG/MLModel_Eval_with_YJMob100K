# ML Model Evaluation with YJMob100K

## Organization

```
|-- Baseline.py                         # Baseline model (haven't fixed the train-test split yet)
|-- CustomTransformer_with_YJMob100K.py # Created a custom Transformer for training YJMob100K data
|-- DataAnalysis_POI.ipynb              # Jupyter Notebook file for POI data analysis
|-- LSTM_with_YJmob100K.py              # Created a built-in LSTM for training YJMob100K data
|-- README.md
|-- res.csv                             # Train and test performance/results of the ML models
|-- dataset.zip                         # contains train.csv and test.csv (10k users' trajectory data)
```

## Run Baseline

```
python Baselin.py
```

## Run LSTM & Transformer in NYU HPC

0. Sign into NYU HPC: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc

1. Clone this repository

```
git clone https://github.com/ANNIZHENG/MLModel_Eval_with_YJMob100K.git
```

2. Request resourses (4 hours, 4 cores, 1 GPU)

```
srun -t 4:00:00 -c 4 --mem=16000 --gres=gpu:4 --pty /bin/bash
```

3. Load CUDA and Python3 packages

**Load desired CUDA package**

```
module spider cuda # optional; check cuda availability
module load cuda/11.6.2
```

**Check Python Availability**

```
module spider python # optional; check python availability
module load python/intel/3.8.6
```

4. Set-up and Activate Virtual Environment: 

```
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
```

5. Install PyTorch

```
pip install torch
pip install dtaidistance
```

6. Go into the repository and start the training

```
cd MLModel_Eval_with_YJMob100K
python CustomTransformer_with_YJMob100K.py # train and test the Transformer
python LSTM_with_YJMob100K.py              # train and test the LSTM
```