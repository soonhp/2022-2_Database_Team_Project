#%%
from scipy.spatial.distance import squareform, pdist
from numpy.linalg import norm
from math import isnan, isinf
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import time
import math
import os
import datetime
import warnings
warnings.filterwarnings(action='ignore')
#%%

# 시간순 정렬

data_name='1m'
default_path = r'D:/OneDrive - 서울과학기술대학교/바탕 화면/21.1/추천시스템공부/코드/order_timestamp_iidup20/' 


sim_name = 'pcc_notime' # 'pcc', 'pcc_notime'


# 수정 세개 해야함

#%%


#Data Load and Preprocessing
if data_name=='100k':
    data = pd.read_csv('C:/RecoSys/Data/u.data', sep='\t', names=['uid','iid','r','ts'], encoding='latin-1')
    data['ts'] = data['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    item_cols=['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation',"Children's",'Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']
    item = pd.read_csv('C:/RecoSys/Data/u.item', sep='|', names=item_cols,encoding='latin-1')
    item=np.array(item.drop(columns=['movie id','movie title', 'release date', 'video release date', 'IMDb URL', 'unknown']))
    # uid, iid minus one. 
    data['uid'] = np.array(data.uid) - 1
    data['iid'] = np.array(data.iid) - 1
    
    

elif data_name=='1m':
    data = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\Movielens Data\ml-1M/ratings.dat', names=['uid','iid','r','ts'],sep='\::',encoding='latin-1',header=None)
    data['ts'] = data['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    item = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\Movielens Data\ml-1M/movies.dat',sep='::',  encoding='latin-1',header=None)
    m_d = {}
    for n, i in enumerate(item.iloc[:,0]):
        m_d[i] = n
    item.iloc[:,0] = sorted(m_d.values())
    
    i_to_n = []
    for i in range(data.shape[0]):
        i_to_n.append(m_d[data.loc[i,'iid']])
    data['iid'] = i_to_n
    
    # uid minus one. 
    data['uid'] = np.array(data.uid) - 1
    #genre matrix
    item = item.set_index(0)
    item = np.array(item.iloc[:,1].str.get_dummies(sep='|'))

    
elif data_name=='amazon_movies':
    data = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\데이터\ratings_Movies_and_TV.csv',names=['uid','iid','r','ts'])
    data['ts'] = data['ts'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    under_20 = data.groupby('iid').size()[data.groupby('iid').size()<20].index
    data = data[data['iid'].isin(under_20) == False]
    data.reset_index(inplace=True,drop=True)
    
    #여기 아래도 한번 손봐야할듯?
    from ast import literal_eval
    item = pd.read_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\데이터/Movies&TV_genre.csv',converters={'category': literal_eval})
    def flatten(l):
        flatList=[]
        for elem in l:
            if type(elem) == list:
                for e in elem:
                    flatList.append(e)
            else:
                flatList.append(elem)
        return flatList
    genre_list = flatten(item['category'])
    genre_unique = np.unique(genre_list)
    
    pd.Series(genre_unique).to_csv(r'D:\OneDrive - 서울과학기술대학교\바탕 화면\21.1\추천시스템공부\데이터/Movies&TV_genre_unique.csv',index=None)
    

    
    
#%%
# Collaborative Filtering
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

# item 기준 sorting

under_20 = data.groupby('iid').size()[data.groupby('iid').size()<20].index
data = data[data['iid'].isin(under_20) == False]
   
   
data.sort_values(by=['iid','ts'],inplace=True)
data.reset_index(drop=True,inplace=True)

iid_count = data['iid'].value_counts().sort_index()


trn_data = pd.DataFrame(columns=data.columns)
val_data = pd.DataFrame(columns=data.columns)

for iid, c in zip(iid_count.index, iid_count):
    trn_data = pd.concat([trn_data,data[data['iid']==iid].iloc[:int(round(c*0.8,0)),:]])
    val_data = pd.concat([val_data,data[data['iid']==iid].iloc[int(round(c*0.8,0)):,:]])
    
trn_data = trn_data.astype({'uid':'int','iid':'int','r':'int','ts':'datetime64[ns]'})
val_data = val_data.astype({'uid':'int','iid':'int','r':'int','ts':'datetime64[ns]'})



# print(len(set(data['iid'])),len(set(trn_data['iid'])),len(set(val_data['iid'])))
# print(len(set(data['uid'])),len(set(trn_data['uid'])),len(set(val_data['uid'])))


#%%

result_mae_rmse = pd.DataFrame(columns=['k','MAE','RMSE'])
result_topN = pd.DataFrame(columns=['k','Precision','Recall','F1_score'])
count = 0

#%%
def sim_pcc(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    
    if len(ind)>1:
        u_m = np.mean(u[ind])
        v_m = np.mean(v[ind])
        pcc = np.sum((u[ind]-u_m)*(v[ind]-v_m)) / (norm(u[ind]-u_m)*norm(v[ind]-v_m)) # range(-1,1)
        if not isnan(pcc): # case: negative denominator
            return pcc
        else:
            return 0
    else:
        return 0
    
def sim_cos(u,v):
    ind=np.where((1*(u!=0)+1*(v!=0))==2)[0]
    if len(ind) > 0:
        up = sum(u[ind] * v[ind])
        down = norm(u[ind]) * norm(v[ind])
        cos_sim = up/down
        if not math.isnan(cos_sim):
            return cos_sim
        else:
            return 0
    else:
        return 0


##########################################################################################

#%%
# train dataset rating dictionary.
n_item = item.shape[0]
n_user = len(set(data['uid']))


data_d_trn_data = {}
for u, i, r in zip(trn_data['uid'], trn_data['iid'], trn_data['r']):
    if i not in data_d_trn_data:
        data_d_trn_data[i] = {u:r}
    else:
        data_d_trn_data[i][u] = r

# train dataset rating dictionary. user
data_d_trn_data_u = {}
data_d_trn_data_t = {}

# user가 train에 다 있지 않음, 빈 딕셔너리값으로 채워줌,,
for _ in range(n_user):
    data_d_trn_data_u[_] = {}
    data_d_trn_data_t[_] = {}

for u, i, r,t in zip(trn_data['uid'], trn_data['iid'], trn_data['r'],trn_data['ts']):
    if data_d_trn_data_u[u] == {}:
        data_d_trn_data_u[u] = {i:r}
        data_d_trn_data_t[u] = {i:t}
    else:
        data_d_trn_data_u[u][i] = r
        data_d_trn_data_t[u][i] = t
    

# train dataset item rating mean dictionary.
data_d_trn_data_mean = {}
for i in data_d_trn_data:
    data_d_trn_data_mean[i] = np.mean(list(data_d_trn_data[i].values()))

### test로 dict 이건 사용자기준 train이랑 transpose느낌~
# user가 test에 다 있지 않음, 빈 딕셔너리값으로 채워줌,,
data_d_tst_data = {}
data_d_tst_data_mean = {}

for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
    if u not in data_d_tst_data:
        data_d_tst_data[u] = {i:r}
    else:
        data_d_tst_data[u][i] = r

for u in data_d_tst_data:
    data_d_tst_data_mean[u] = np.mean(list(data_d_tst_data[u].values()))
###


# train rating matrix
rating_matrix = np.zeros((n_user, n_item))
time_matrix = pd.DataFrame(np.zeros((n_user, n_item)))
for u, i, r, t in zip(trn_data['uid'], trn_data['iid'], trn_data['r'], trn_data['ts']):
    rating_matrix[u,i] = r
    time_matrix.iloc[u,i] = t

time_matrix = np.array(time_matrix)

# test rating matrix
rating_matrix_test = np.zeros((n_user, n_item))
# time_matrix_test = np.zeros((n_user, n_item))
data_d_tst_data_t = {}
for u, i, r, t in zip(val_data['uid'], val_data['iid'], val_data['r'], val_data['ts']):
    rating_matrix_test[u,i] = r
    # time_matrix_test[u,i] = t
    if u not in data_d_tst_data_t:
        data_d_tst_data_t[u] = {i:t}
    else:
        data_d_tst_data_t[u][i] = t
        
#%% sparsity 계산

# df = data.pivot(index='uid', columns='iid', values='r')
# df.fillna(0,inplace=True)

# print(sum(np.array(df).reshape(-1,) == 0) / len(np.array(df).reshape(-1,)))



#%%
#유사도계산#############################################################################################      
print('\n')
print(f'similarity calculation: {sim_name}')



if sim_name=='cos_notime':    
    sim=pdist(rating_matrix.T,metric=sim_cos)
    np.save(default_path + '/sim_result/cos_notime_sim_'+data_name,sim)
    # sim=np.load(default_path + '/sim_result/cos_notime_sim_'+data_name+'.npy')
    sim=squareform(sim)


   
elif sim_name=='pcc_notime':    
    sim=pdist(rating_matrix.T,metric=sim_pcc)
    np.save(default_path + '/sim_result/pcc_notime_sim_'+data_name,sim)
    # sim=np.load(default_path + '/sim_result/pcc_notime_sim_'+data_name+'.npy')
    sim=squareform(sim)
    


    
elif sim_name == 'acos_notime':
    rating_T_mean = rating_matrix.T - rating_matrix.T.mean(axis=0)
    sim = []
    for x in range(rating_matrix.shape[1]):
        for y in range(x+1,rating_matrix.shape[1]):
            u = rating_T_mean[x,:]
            v = rating_T_mean[y,:]
            u1 = rating_matrix[:,x]
            v1 = rating_matrix[:,y]
            ind=np.where((1*(u1!=0)+1*(v1!=0))==2)[0]
            if len(ind)>0:
                acos_sim = sum(u[ind] * v[ind]) / (norm(u[ind]) * norm(v[ind]))
                
                if isnan(acos_sim):
                    acos_sim = 0
            else:
                acos_sim = 0
            sim.append(acos_sim)
    sim = np.array(sim)
    np.save(default_path + '/sim_result/acos_notime_sim_'+data_name,sim)
    sim=squareform(sim)



#%%


np.fill_diagonal(sim,-100) # 정렬하기 전..! -1값 다른거로 바꿔도 될듯!?
nb_ind=np.argsort(sim,axis=1)[:,::-1] # nearest neighbor sort, 1행 : item 0의  index,, 즉 -> 이 방향
sel_nn=nb_ind[:,:100] # 상위100명
sel_sim=np.sort(sim,axis=1)[:,::-1]




print('\n')
print('prediction: k=10,20, ..., 100')
rating_matrix_prediction = rating_matrix.copy()
    
s=time.time()
e=0

###
# val_data = val_data.sort_values(by='uid')
# uids = sorted(list(set(data['uid'])))

###
for k in tqdm([10,20,30,40,50,60,70,80,90,100]):
    
    for user in range(rating_matrix.shape[0]): # user를 돌고, # 
        
        for p_item in list(np.where(rating_matrix_test[user,:]!=0)[0]): # 예측할 item을 돌고, p_item=252
            
            molecule = []
            denominator = []
            
            # call K neighbors 아이템 p_item 이랑 유사한 k개의 아이템들이 item_neihbor이 되고,,
            item_neighbor = sel_nn[p_item,:k]
            item_neighbor_sim = sel_sim[p_item,:k]

            for neighbor, neighbor_sim in zip(item_neighbor, item_neighbor_sim): # neighbor=337
                if neighbor in data_d_trn_data_u[user].keys():
                    molecule.append(neighbor_sim * (rating_matrix[user, neighbor] - data_d_trn_data_mean[neighbor]))
                    denominator.append(abs(neighbor_sim))
            try:
                rating_matrix_prediction[user, p_item] = data_d_trn_data_mean[p_item] + (sum(molecule) / sum(denominator))
            except : #ZeroDivisionError: user가 p_item의 이웃item을 평가한 적이 없는 경우, KeyError: test에는 있는데 train에는 없는 item.
                e+=1
                rating_matrix_prediction[user, p_item] = math.nan
              
                '''
                a : p_item
                i : neighbor
                u : user
                w_ai : neighbor sim
                '''
              
                
   #3. performance
        # MAE, RMSE
        
    precision, recall, f1_score = [], [], []
    # 평균!!
    rec_score = 4 # 추천 기준 점수.
    pp=[]
    rr=[]
    n_tp = 0
    n_fp = 0
    n_fn = 0
    for u, i, r in zip(val_data['uid'], val_data['iid'], val_data['r']):
        p = rating_matrix_prediction[u,i]
        if not math.isnan(p):
            pp.append(p) # 예측
            rr.append(r) # 실제
            u_mean = data_d_tst_data_mean[u]
            if p >= u_mean and r >= u_mean:
                n_tp += 1
            elif p >= u_mean and r < u_mean:
                n_fp += 1
            elif p < u_mean and r >= u_mean:
                n_fn += 1
    _precision = n_tp / (n_tp + n_fp)
    _recall = n_tp / (n_tp + n_fn)
    _f1_score = 2 * _precision * _recall / (_precision + _recall)      
            

    
    d = [abs(a-b) for a,b in zip(pp,rr)]
    mae = sum(d)/len(d)
    rmse = np.sqrt(sum(np.square(np.array(d)))/len(d))
    
    result_mae_rmse.loc[count] = [k, mae, rmse]


    result_topN.loc[count] = [k, _precision, _recall, _f1_score]
    count += 1

result_1 = result_mae_rmse.groupby(['k']).mean()
result_2 = result_topN.groupby(['k']).mean()
result = pd.merge(result_1, result_2, on=result_1.index).set_index('key_0')
print(result)


result.to_csv(default_path + '/result_f1/'+ sim_name +'&'+data_name+'.csv')


    
