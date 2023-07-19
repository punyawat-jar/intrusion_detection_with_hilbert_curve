#Training set and Testing set wil be processed in this file
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import requests, json

from multiprocessing import Pool, cpu_count

def send_discord_message(content):
    webhook_url = 'https://discord.com/api/webhooks/1120592724123467796/KLwL2pWifliFuwOzs_Az-VSGk8n1fz2lSEEEUsmEZ-UFgpRfWXHPmlZXW3AFhsmY4FWU'

    data = {
        'content': content
    }

    response = requests.post(webhook_url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    if response.status_code != 204:
        raise ValueError(f'Request to discord returned an error {response.status_code}, the response is:\n{response.text}')


def read_newDB_test():
    feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    dftest = pd.read_csv('/home/s2316002/Project2/Dataset/KDDTest+.txt',names = feature)
    dftest.to_csv('/home/s2316002/Project2/Dataset/KDDTest+.csv', index =False)
    return dftest

def read_newDB_train():
    feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

    dftrain = pd.read_csv('/home/s2316002/Project2/Dataset/KDDTrain+.txt',names = feature)
    dftrain.to_csv('/home/s2316002/Project2/Dataset/KDDTrain+.csv', index =False)
    return dftrain

def preprocess_test(df):
    cat_cols = ['is_host_login','protocol_type','service','flag','land', 'logged_in','is_guest_login', 'label']

    df.drop(['difficulty'],axis=1,inplace=True)
    
    df_num = df.drop(cat_cols, axis=1)
    
    df.loc[df['label'] == "normal", "label"] = 0
    df.loc[df['label'] != 0, "label"] = 1
    
    df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'], dtype='int')
    
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])
    

    return df

def preprocess_train(dataframe):
    dataframe.drop(['difficulty'],axis=1,inplace=True)
    
    dataframe.loc[dataframe['label'] == "normal", "label"] = 0
    dataframe.loc[dataframe['label'] != 0, "label"] = 1
    # dataframe = dataframe.drop(['label'],axis = 1)
    dataframe = pd.get_dummies(dataframe, columns=['protocol_type', 'service', 'flag'], dtype='int')
    
    scaler = MinMaxScaler()
    dataframe[dataframe.columns] = scaler.fit_transform(dataframe[dataframe.columns])
    return dataframe

def align_columns(dataset1, dataset2, column_sequence):
    # Reindex both datasets with the aligned column order
    dataset1 = dataset1.reindex(columns=column_sequence)
    dataset2 = dataset2.reindex(columns=column_sequence)

    # Set NaN values to 0
    dataset1 = dataset1.fillna(0)
    dataset2 = dataset2.fillna(0)

    return dataset1, dataset2

def check_column_differences(dataset1, dataset2):
    columns1 = set(dataset1.columns)
    columns2 = set(dataset2.columns)

    columns_only_in_dataset1 = columns1 - columns2
    columns_only_in_dataset2 = columns2 - columns1

    return columns_only_in_dataset1, columns_only_in_dataset2


def save_plot(args):
    i, seq, df_name = args
    dpi = 80
    height, width = seq.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(seq, cmap='binary', aspect='auto')
    plt.axis('off')  # Turn off the axis
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)  # Remove padding and margin
    if df_name == 'train':
        plt.savefig(f'/home/s2316002/Project2/netflow/TrainImages/image_{i}.png', bbox_inches = 'tight',
            pad_inches = 0)  # Save without padding
    else: # df_name == 'test'
        plt.savefig(f'/home/s2316002/Project2/netflow/TestImages/image_{i}.png', bbox_inches = 'tight',
            pad_inches = 0)  # Save without padding
    plt.close(fig)  # Close the figure to free up memory

def create_sequences(df, window_size):
    sequences = []
    for i in range(len(df) - window_size + 1):
        sequences.append(df.iloc[i:i+window_size, :])
    return sequences

send_discord_message('== Starting netflow ==')

dftrain = read_newDB_train()
dftest = read_newDB_test()

dftrain = preprocess_train(dftrain)
dftest = preprocess_test(dftest)

dftrain, dftest = align_columns(dftrain, dftest, dftrain.columns)
m, n = check_column_differences(dftrain, dftest)
dftrain = dftrain.drop('label', axis=1)
dftest = dftest.drop('label', axis=1)
send_discord_message('== Creating sequence ==')
window_size = 64  # Define your window size
sequences_train = create_sequences(dftrain, window_size)
sequences_test = create_sequences(dftest, window_size)

# Get the number of available CPUs for multiprocessing
num_cpus = cpu_count()
send_discord_message(f'cpus : {num_cpus}')

# Create a multiprocessing Pool
pool = Pool(num_cpus)

# Use the pool to generate the images in parallel
pool.map(save_plot, [(i, seq, 'train') for i, seq in enumerate(sequences_train)])
pool.map(save_plot, [(i, seq, 'test') for i, seq in enumerate(sequences_test)])

# Close the pool to free up resources
pool.close()
pool.join()

send_discord_message('== Preprocessing completed ==')
