# IDS-TOOL
from google.colab import drive
drive.mount('/content/drive')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
with open("/content/drive/MyDrive/colab_dataset/Intrusion/kddcup.names",'r') as f:
    print(f.read())
cols="""duration,
protocol_type,
service,
flag,
src_bytes,
dst_bytes,
land,
wrong_fragment,
urgent,
hot,
num_failed_logins,
logged_in,
num_compromised,
root_shell,
su_attempted,
num_root,
num_file_creations,
num_shells,
num_access_files,
num_outbound_cmds,
is_host_login,
is_guest_login,
count,
srv_count,
serror_rate,
srv_serror_rate,
rerror_rate,
srv_rerror_rate,
same_srv_rate,
diff_srv_rate,
srv_diff_host_rate,
dst_host_count,
dst_host_srv_count,
dst_host_same_srv_rate,
dst_host_diff_srv_rate,
dst_host_same_src_port_rate,
dst_host_srv_diff_host_rate,
dst_host_serror_rate,
dst_host_srv_serror_rate,
dst_host_rerror_rate,
dst_host_srv_rerror_rate"""

columns=[]
for c in cols.split(','):
    if(c.strip()):
       columns.append(c.strip())

columns.append('target')
#print(columns)
print(len(columns))
with open("/content/drive/MyDrive/colab_dataset/Intrusion/training_attack_types",'r') as f:
    print(f.read())
attacks_types = {
    'normal': 'normal',
'back': 'dos',
'buffer_overflow': 'u2r',
'ftp_write': 'r2l',
'guess_passwd': 'r2l',
'imap': 'r2l',
'ipsweep': 'probe',
'land': 'dos',
'loadmodule': 'u2r',
'multihop': 'r2l',
'neptune': 'dos',
'nmap': 'probe',
'perl': 'u2r',
'phf': 'r2l',
'pod': 'dos',
'portsweep': 'probe',
'rootkit': 'u2r',
'satan': 'probe',
'smurf': 'dos',
'spy': 'r2l',
'teardrop': 'dos',
'warezclient': 'r2l',
'warezmaster': 'r2l',
}

attacks_types
READING DATASET
path = "/content/drive/MyDrive/colab_dataset/Intrusion/kddcup.data_10_percent.gz"
df = pd.read_csv(path,names=columns)

#Adding Attack Type column
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]])
df.shape
df['target'].value_counts()
df['Attack Type'].value_counts()
df.dtypes
DATA PREPROCESSING
df.isnull().sum()
#Finding categorical features
num_cols = df._get_numeric_data().columns

cate_cols = list(set(df.columns)-set(num_cols))
cate_cols.remove('target')
cate_cols.remove('Attack Type')

cate_cols
CATEGORICAL FEATURES DISTRIBUTION
#Visualization
def bar_graph(feature):
    df[feature].value_counts().plot(kind="bar")
bar_graph('protocol_type')
Protocol type: We notice that ICMP is the most present in the used data, then TCP and almost 20000 packets of UDP type
plt.figure(figsize=(15,3))
bar_graph('service')
bar_graph('flag')
bar_graph('logged_in')
logged_in (1 if successfully logged in; 0 otherwise): We notice that just 70000 packets are successfully logged in.
TARGET FEATURE DISTRIBUTION
bar_graph('target')
Attack Type(The attack types grouped by attack, it's what we will predict)
bar_graph('Attack Type')
df.columns
DATA CORRELATION
df = df.dropna('columns')# drop columns with NaN

df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values

corr = df.corr()

plt.figure(figsize=(15,12))

sns.heatmap(corr)

plt.show()
df['num_root'].corr(df['num_compromised'])
df['srv_serror_rate'].corr(df['serror_rate'])
df['srv_count'].corr(df['count'])
df['srv_rerror_rate'].corr(df['rerror_rate'])
df['dst_host_same_srv_rate'].corr(df['dst_host_srv_count'])
df['dst_host_srv_serror_rate'].corr(df['dst_host_serror_rate'])
df['dst_host_srv_rerror_rate'].corr(df['dst_host_rerror_rate'])
df['dst_host_same_srv_rate'].corr(df['same_srv_rate'])
df['dst_host_srv_count'].corr(df['same_srv_rate'])
df['dst_host_same_src_port_rate'].corr(df['srv_count'])
df['dst_host_serror_rate'].corr(df['serror_rate'])
df['dst_host_serror_rate'].corr(df['srv_serror_rate'])
df['dst_host_srv_serror_rate'].corr(df['serror_rate'])
df['dst_host_srv_serror_rate'].corr(df['srv_serror_rate'])
df['dst_host_rerror_rate'].corr(df['rerror_rate'])
df['dst_host_rerror_rate'].corr(df['srv_rerror_rate'])
df['dst_host_srv_rerror_rate'].corr(df['rerror_rate'])
df['dst_host_srv_rerror_rate'].corr(df['srv_rerror_rate'])
#This variable is highly correlated with num_compromised and should be ignored for analysis.
#(Correlation = 0.9938277978738366)
df.drop('num_root',axis = 1,inplace = True)

#This variable is highly correlated with serror_rate and should be ignored for analysis.
#(Correlation = 0.9983615072725952)
df.drop('srv_serror_rate',axis = 1,inplace = True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9947309539817937)
df.drop('srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_serror_rate and should be ignored for analysis.
#(Correlation = 0.9993041091850098)
df.drop('dst_host_srv_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9869947924956001)
df.drop('dst_host_serror_rate',axis = 1, inplace=True)

#This variable is highly correlated with srv_rerror_rate and should be ignored for analysis.
#(Correlation = 0.9821663427308375)
df.drop('dst_host_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with rerror_rate and should be ignored for analysis.
#(Correlation = 0.9851995540751249)
df.drop('dst_host_srv_rerror_rate',axis = 1, inplace=True)

#This variable is highly correlated with dst_host_srv_count and should be ignored for analysis.
#(Correlation = 0.9865705438845669)
df.drop('dst_host_same_srv_rate',axis = 1, inplace=True)
df.head()
df.shape
df.columns
df_std = df.std()
df_std = df_std.sort_values(ascending = True)
df_std
FEATURE MAPPING
df['protocol_type'].value_counts()
#protocol_type feature mapping
pmap = {'icmp':0,'tcp':1,'udp':2}
df['protocol_type'] = df['protocol_type'].map(pmap)
df['flag'].value_counts()
#flag feature mapping
fmap = {'SF':0,'S0':1,'REJ':2,'RSTR':3,'RSTO':4,'SH':5 ,'S1':6 ,'S2':7,'RSTOS0':8,'S3':9 ,'OTH':10}
df['flag'] = df['flag'].map(fmap)
df.head()
df.drop('service',axis = 1,inplace= True)
df.shape
df.head()
df.dtypes
MODELLING
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
df = df.drop(['target',], axis=1)
print(df.shape)

# Target variable and train set
Y = df[['Attack Type']]
X = df.drop(['Attack Type',], axis=1)

sc = MinMaxScaler()
X = sc.fit_transform(X)

# Split test and train data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)
RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30)
model.fit(X_train, Y_train.values.ravel())
#[0,1,0,181,5450,0,0,0,0,0,1,0,0,0,0,0,0,0,8,8,0,0,1,0,0,9,9,0,0.11,0]
#[0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,18,1,0,0.08,0.06,0,255,18,0.07,0,0]
arr = np.array([0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,216,18,1,0,0.08,0.06,0,255,18,0.07,0,0])
rf_pred = model.predict(arr.reshape(1,-1))
rf_pred[0]
print("Train score is:", model.score(X_train, Y_train))
print("Test score is:",model.score(X_test,Y_test))
import joblib
joblib.dump(model, "model.h5")
