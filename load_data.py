import gzip
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# Load dataset under normal scenes
yjmob1 = 'yjmob100k-dataset1.csv.gz'
yjmob_df = pd.read_csv(yjmob1, compression='gzip')

# Retrieve all unique uids
uids = sorted(yjmob_df['uid'].unique())

# Randomly select 10k users
selected_uids = random.sample(uids, 10000)
df = yjmob_df[yjmob_df['uid'].isin(selected_uids)]

# Create grid-like location-id via linearization
def spatial_token(x, y):
    """
    x: int
    y: int
    ret: gird-id mapped from (x,y)
    """
    return (x-1)+(y-1)*200

df = df.copy() # this is to suppress SettingWithCopyWarning
df['combined_xy'] = df.apply(lambda row: spatial_token(row['x'], row['y']), axis=1)

# 8:2 train-test split of uids

# horizontal split
# train_uids, test_uids = train_test_split(selected_uids, test_size=0.2, random_state=123)
# df_train = df[df['uid'].isin(train_uids)]
# df_test = df[df['uid'].isin(test_uids)]

# veritcal split
train_data = []
test_data = []
for uid, group in df.groupby('uid'):
    # if len(group) >= 1000:
    selected_data = group.head(1000)
    train_rows = selected_data.head(800) # 800 train data
    test_rows = selected_data.iloc[800:1000] # 200 test data
    train_data.append(train_rows)
    test_data.append(test_rows)

df_train = pd.concat(train_data)
df_test = pd.concat(test_data)

# Export data for model use
df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)