import gzip
import pandas as pd
import random

yjmob1 = 'yjmob100k-dataset1.csv.gz' # dataset under normal scenes
yjmob_df = pd.read_csv(yjmob1, compression='gzip')
uids = yjmob_df['uid'].unique()

rand_indicies = [random.randint(0, len(uids)) for _ in range(10000)]
selected_uids = [uid for uid in uids[rand_indicies]]
df = yjmob_df[yjmob_df['uid'].isin(selected_uids)] 

# Location Data
# linearization of the 2-dimensional grid, i.e., the original x,y coordinate system
def spatial_token(x, y):
    # x,y are the coordinate location
    # x determines the column order while
    # y determines the row order
    # (x-1) calculates the starting grid-column position
    # (y-1)*200 calculates the start index of the grid-row
    return (x-1)+(y-1)*200
df['combined_xy'] = df.apply(lambda row: spatial_token(row['x'], row['y']), axis=1)

# 8:2 split of ID
train_uids, test_uids = train_test_split(selected_uids, test_size=0.2, random_state=42)
df_train = df[df['uid'].isin(train_uids)]
df_test = df[df['uid'].isin(test_uids)]

# export data
df_train.to_csv('df_train.csv', index=False)
df_test.to_csv('df_test.csv', index=False)