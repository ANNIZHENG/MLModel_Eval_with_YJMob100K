# ML Model Evaluation with YJMob100K

## Organization

```
|-- CustomTransformer_with_YJMob100K.py # Created a custom Transformer for training YJMob100K data
|-- LSTM_with_YJmob100K.py              # Created a built-in LSTM for training YJMob100K data
|-- train.csv                           # 8K users each with 150 step data
|-- test.csv                            # 2k users each with 150 step data
|-- res.csv                             # Train and test performance/results of the ML models
|-- README.md
```

## Set up cluster training in NYU HPC

0. Sign into NYU HPC: https://sites.google.com/nyu.edu/nyu-hpc/accessing-hpc

1. Clone this repository

`git clone https://github.com/ANNIZHENG/MLModel_Eval_with_YJMob100K.git`

2. Request resourses

`srun -t 4:00:00 -c 4 --mem=16000 --gres=gpu:1 --pty /bin/bash` (4 hours and 1 GPU)

3. Load CUDA and Python3 packages

3.1 Check CUDA Availability

`module spider cuda`

3.2 Load desired CUDA package

`module load cuda/11.6.2` (this is the newest version in my case)

3.3 Check Python Availability

`module spider python`

3.4 Load desired Python package

`module load python/intel/3.8.6` (the newest version in my case)

4. Set-up and Activate Virtual Environment: 

```
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate
```

5. Import PyTorch

`pip install torch`

6. Go into the repository and start the training

```
cd MLModel_Eval_with_YJMob100K
python CustomTransformer_with_YJMob100K.py # train and test the Transformer
python LSTM_with_YJMob100K.py              # train and test the LSTM
```