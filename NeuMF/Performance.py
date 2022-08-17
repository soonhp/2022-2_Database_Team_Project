# -*- coding: utf-8 -*-
"""
Created on Wed May 18 19:06:03 2022

@author: user
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
data_path = r'D:/OneDrive - 서울과학기술대학교/바탕 화면/22.1/데이터베이스시스템/팀프로젝트/'

def get_files_count(folder_path):
	dirListing = os.listdir(folder_path)
	return dirListing

file_names = get_files_count(data_path+'runs_neumf/')

#%%


epoch = []
hr = []
ndcg = []

for file_name in file_names:
    epoch.append(int(file_name.split('_')[3][5:])) # 2/7/3
    hr.append(float(file_name.split('_')[4][2:])) # 3/8/4
    ndcg.append(float(file_name.split('_')[5][4:10])) # 4/9/5
    
performance = pd.DataFrame({'epoch':epoch, 'HR' :hr, 'NDCG' : ndcg})

performance = performance.sort_values('epoch')
performance.reset_index(drop=True,inplace=True)

performance.to_csv(data_path+'neumf_pf.csv',index=None)

#%%

gmf = pd.read_csv(data_path + 'gmf_pf.csv')
mlp = pd.read_csv(data_path + 'mlp_pf.csv')
neumf = pd.read_csv(data_path + 'neumf_pf.csv')



#%% Plotting

plt.plot(gmf['HR'],label = 'MF')
plt.plot(mlp['HR'], label = 'MLP(pretrain)')
plt.plot(neumf['HR'], label = 'NeuMF(pretrain)')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('hit ratio@10')
plt.show()



plt.plot(gmf['NDCG'],label = 'MF')
plt.plot(mlp['NDCG'], label = 'MLP(pretrain)')
plt.plot(neumf['NDCG'], label = 'NeuMF(pretrain)')
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('NDCG@10')
plt.show()

#%%
