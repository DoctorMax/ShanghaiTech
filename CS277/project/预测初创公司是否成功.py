#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 时间：2021年5月22日14:22:22
# 贡献者：刘
# 前置文件：test8.ipynb
# 文件内容：这是参考文献1的原始代码修改
# 文件描述：功能没有太大变化，依然是数据处理然后逻辑回归
# 实现内容：把代码修修漂亮，这是最终的提交版
# TBD：NaN


# # 初始化

# In[2]:


# 导入依赖包和一些初始化
import re
import csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# 画图用的
import seaborn as sns
# 设置浮点数格式
pd.options.display.float_format = '{:.4f}'.format
import datetime as dt
import matplotlib.dates as mdates
# 画图不弹窗，直接内联显示
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')
# 让代码忽略警告，继续运行
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.utils import resample
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.feature_selection import SelectKBest
from matplotlib import rcParams
from sklearn.utils import resample
# 画图的时候自适应布局
rcParams.update({'figure.autolayout': True})

# 一个数据分析的神仙库
import pandas_profiling as pp


# # 读数据

# In[3]:


# 加载数据，编码格式是mac_roman
crunchbase = pd.read_csv("cb_objects.csv",index_col=0, encoding='mac_roman')
people = pd.read_csv("cb_people.csv")
ipoInfo=pd.read_csv("cb_ipos.csv")
acquisitionsInfo=pd.read_csv("cb_acquisitions.csv")
degreesInfo=pd.read_csv("cb_degrees.csv")


# In[4]:


# 看一眼前五行有啥
crunchbase.head()
# 企业名字，企业主页，企业所处市场类型，融资金额，经营状态（acquired收购，operating运营），所处国家，所处州，所处区域，所处城市，
# 融资轮数，成立日期，成立月份，成立季度，成立年份，首次融资日期，最后融资日期


# In[5]:


crunchbase.shape


# In[6]:


# 看一眼创业者的数据有啥
people.head()
# 人的id，人的类型，名，姓，数据库中的链接，照片链接，facebook链接，推特链接，领英链接，所在城市，地区，国家，职称，所属组织（公司的母公司），链接


# In[7]:


ipoInfo.head()


# In[8]:


acquisitionsInfo.head()


# In[9]:


degreesInfo.head()


# # 数据处理

# ## 洗公司数据

# In[10]:


# 重建一个索引
crunchbase.reset_index(inplace=True)


# In[11]:


# 挑选出我们可能用到的公司数据
crunchbase=crunchbase.loc[crunchbase["entity_type"]=="Company"]
crunchbase=crunchbase.drop(["entity_type", 
                 "entity_id",
                 "parent_id",
                 "normalized_name",
                 "permalink",
                 "closed_at", 
                 "domain" ,
                 "homepage_url",
                 "twitter_username",
                 "logo_url",
                 "logo_width",
                 "logo_height",
                 "short_description",
                 "description",
                 "overview",
                 "tag_list",
                 "state_code",
                 "region",
                 "first_investment_at",
                 "last_investment_at",
                 "investment_rounds",
                 "invested_companies",
                 "first_funding_at",
                 "last_funding_at", 
                 "first_milestone_at",
                 "last_milestone_at", 
                 "milestones",
                 "created_by",
                 "created_at",
                 "updated_at"], axis=1)


# In[12]:


crunchbase.head()


# In[13]:


crunchbase.shape


# In[14]:


# 针对代码进行一些列名替换
crunchbase=crunchbase.rename(columns={'category_code':'market'})


# In[15]:


# # 把轮数中非数字删掉，空值保留（如果没有非数字类型则会报错）
# crunchbase.drop(crunchbase[~crunchbase.funding_rounds.str.match('^[0-9]+$',na=True)].index,inplace=True)


# In[16]:


# # 把关系数中非数字删掉，空值保留
# crunchbase.drop(crunchbase[~crunchbase.relationships.str.match('^[0-9]+$',na=True)].index,inplace=True)


# In[17]:


# 把总资金中非数字删掉，空值赋0
crunchbase['funding_total_usd']=crunchbase['funding_total_usd'].fillna('0')
# crunchbase.drop(crunchbase[~crunchbase.funding_total_usd.str.match('^[0-9]+$')].index,inplace=True)
crunchbase['funding_total_usd'] = crunchbase['funding_total_usd'].astype(float)


# In[18]:


# 把日期中的其他东西删掉，空值赋0
crunchbase['founded_at']=crunchbase['founded_at'].fillna('0/0/0')
crunchbase.drop(crunchbase[~crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=True)].index,inplace=True)


# In[19]:


# 把投资轮数中非数字删掉
# crunchbase.drop(crunchbase[~crunchbase.funding_rounds.str.match('^[0-9]+$',na=True)].index,inplace=True)
# crunchbase['funding_rounds'] = crunchbase['funding_rounds'].astype(float)


# In[20]:


# 提取日期中的年份信息，！！请不要修改，套娃套太多了，看不懂的
crunchbase['founded_year']=np.array(crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/').values.tolist())[:,2]
# 删掉0
crunchbase['founded_year'].str.strip('^0$')
crunchbase['founded_year'] = crunchbase['founded_year'].astype(float)
# 帮助理解这块代码的一些参数
# crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)
# crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)]
# crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/')
# crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/').values
# crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/').values.tolist()
# np.array(crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/').values.tolist())
# np.array(crunchbase[crunchbase.founded_at.str.match('\d+\/+\d+\/+\d',na=False)].founded_at.str.split('/').values.tolist())[:,2]


# In[21]:


# 把原本企业创建日期删掉
crunchbase=crunchbase.drop("founded_at",axis=1)


# In[22]:


crunchbase.shape


# ## 洗创始人信息

# In[23]:


people=people[["object_id","affiliation_name"]]
people=people.dropna(axis=0,subset = ["object_id","affiliation_name"])
people=people.drop_duplicates(subset='object_id')


# ## 洗IPO数据

# In[24]:


ipoInfo=ipoInfo[["object_id","public_at"]]
ipoInfo=ipoInfo.dropna(axis=0,subset = ["object_id","public_at"])
ipoInfo=ipoInfo.drop_duplicates(subset='object_id')
ipoInfo['public_at']=np.array(ipoInfo[ipoInfo.public_at.str.match('\d+\/+\d+\/+\d',na=False)].public_at.str.split('/').values.tolist())[:,2]
ipoInfo['public_at'].str.strip('^0$')
ipoInfo['public_at'] = ipoInfo['public_at'].astype(float)


# ## 洗收购信息

# In[25]:


acquisitionsInfo=acquisitionsInfo[["acquired_object_id","acquired_at"]]
acquisitionsInfo=acquisitionsInfo.dropna(axis=0,subset = ["acquired_object_id","acquired_at"])
acquisitionsInfo=acquisitionsInfo.drop_duplicates(subset='acquired_object_id')
acquisitionsInfo['acquired_at']=np.array(acquisitionsInfo[acquisitionsInfo.acquired_at.str.match('\d+\/+\d+\/+\d',na=False)].acquired_at.str.split('/').values.tolist())[:,2]
acquisitionsInfo['acquired_at'].str.strip('^0$')
acquisitionsInfo['acquired_at'] = acquisitionsInfo['acquired_at'].astype(float)


# ## 洗学位信息

# In[26]:


degreesInfo=degreesInfo[["object_id","degree_type"]]
degreesInfo=degreesInfo.dropna(axis=0,subset = ["object_id","degree_type"])
degreesInfo=degreesInfo.drop_duplicates(subset='object_id')
degreesInfo['degree_type'][(degreesInfo['degree_type']!='BS')
                           & (degreesInfo['degree_type']!='MBA')
                           &(degreesInfo['degree_type']!='BA')
                           &(degreesInfo['degree_type']!='MS')
                           &(degreesInfo['degree_type']!='PhD')]='unknown'


# In[27]:


degreesInfo.head()


# In[28]:


degreesInfoOnehot = pd.get_dummies(degreesInfo['degree_type'])
degreesInfo=pd.merge(degreesInfo,degreesInfoOnehot,left_index= True ,right_index= True)
degreesInfo=degreesInfo.drop('degree_type',axis=1)


# In[29]:


degreesInfo.head()


# ## 融合后数据处理

# In[30]:


# 先把创始人信息和学位融合
master = pd.merge(people, degreesInfo, how='left',left_on=['object_id'], right_on=['object_id'])
master['BA']=master['BA'].fillna(0)
master['BS']=master['BS'].fillna(0)
master['MBA']=master['MBA'].fillna(0)
master['MS']=master['MS'].fillna(0)
master['PhD']=master['PhD'].fillna(0)
master['unknown']=master['unknown'].fillna(1)


# In[31]:


master.head()


# In[32]:


# 融合创业者和初创企业的关系为一个数据
master=pd.merge(master, crunchbase, how='left',left_on=['affiliation_name'], right_on=['name'])
master=master.drop('affiliation_name',axis=1)


# In[33]:


master.head()


# In[34]:


# 统计创业者的任职公司的分布，看哪家公司出来几个人
peopleSum=master.groupby(['name']).agg({'object_id': "nunique", 'BA':'sum',  'BS':'sum', 'MBA':'sum', 'MS':'sum', 'PhD':'sum','unknown':'sum'})
peopleSum=peopleSum.rename(columns = {"object_id": "numEntrepreneurs"})
# 重建一个索引
peopleSum = peopleSum.reset_index()


# In[35]:


peopleSum.head()


# In[36]:


master=pd.merge(crunchbase, peopleSum, how='right',left_on=['name'], right_on=['name'])


# In[37]:


master.head()


# In[38]:


# 再把上市时间融合进去
master=pd.merge(master, ipoInfo,how='left',left_on=['id'],right_on=['object_id']).drop("object_id",axis=1)


# In[39]:


# 再把被收购时间融合进去
master=pd.merge(master, acquisitionsInfo,how='left',left_on=['id'],right_on=['acquired_object_id']).drop("acquired_object_id",axis=1)


# In[40]:


# 检查一下数据集大小，确保融合是正确的
print(crunchbase.shape)
print(people.shape)
print(ipoInfo.shape)
print(acquisitionsInfo.shape)
print(master.shape)


# In[41]:


master.head()


# In[42]:


# 企业成功的定义！对企业的运营状况onehot特征化

master['success']=0
master['success'][(master['status'] == 'ipo') & (master['public_at']-master['founded_year']<=5)]=1
master['success'][(master['status'] == 'acquired') & (master['acquired_at']-master['founded_year']<=5)]=1


# In[43]:


master.success.sum()


# ## 只关注2000年金融危机后的数据

# In[44]:


# 根据2000年金融危机，所以我们定义2000年以后成立的企业是初创公司，第一次数据锐减，从196553~76432
master['startup'] = 0
master['startup'][(master['founded_year'] > 2000)] = 1
startups = master[(master['startup'] == 1)]


# In[45]:


startups.shape


# In[46]:


startups.head()


# In[47]:


startups.success.sum()


# In[48]:


# 我们的数据是到2013年截止的，计算运营时间
startups['age'] = 2013 - startups['founded_year']


# In[49]:


# 保存数据
startups.to_csv('startups.csv')


# # 数据分析

# In[50]:


# 对数据类型是数字的特征（包括刚特征化的企业运营状况）计算相关系数
master.corr()


# In[51]:


# 统计一下刚才这些数字特征
master.describe().transpose()


# In[52]:


# 统计一下初创公司的数值信息
startups.describe().transpose()


# # 画初创企业的图

# ##  画出初创企业的前10个分布国家

# In[53]:


# 数据处理
countries_startups = startups.groupby(['country_code']).agg({'name': 'nunique',
                                                    'success': "sum"}).sort_values(by="name",ascending=False)

countries_startups['fail'] = countries_startups['name'] - countries_startups['success']
countries_startups['success_per'] = countries_startups['success'] / countries_startups['name']

countries_startups=countries_startups.head(10)
countries_startups.reset_index(inplace=True)
countries_startups


# In[54]:


# 画图
labels = countries_startups.country_code
success_means = countries_startups.success
fail_means = countries_startups.fail
y2=countries_startups.success_per


width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.bar(labels,success_means, width, label='success')
ax.bar(labels, fail_means, width, bottom=success_means,
       label='Fail')
ax.set_xlabel('Country')
ax.set_ylabel('Startups Num')


ax2.set_ylabel('success Rate', color='g')
ax2.plot(labels, y2, 'g--')
ax.set_title('Number of Startups per Country')
ax.legend()
# fig.show()
fig.savefig('Number_of_Startups_per_Country.png', dpi=1000, transparent=True)


# ## 画出初创企业的前十个分布城市

# In[55]:


# 数据处理
cities_startups = startups.groupby(['city']).agg({'name': 'nunique',
                                                    'success': "sum"}).sort_values(by="name",ascending=False)

cities_startups['fail'] = cities_startups['name'] - cities_startups['success']
cities_startups['success_per'] = cities_startups['success'] / cities_startups['name']

cities_startups=cities_startups.head(10)
cities_startups.reset_index(inplace=True)
cities_startups


# In[56]:


# 画图
labels = cities_startups.city
success_means = cities_startups.success
fail_means = cities_startups.fail
y2=cities_startups.success_per


width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.bar(labels,success_means, width, label='success')
ax.bar(labels, fail_means, width, bottom=success_means,
       label='Fail')
ax.set_xlabel('City')
ax.set_ylabel('Startups Num')


ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") # ha表示文字的哪个位置是对应坐标轴的位置参考

ax2.set_ylabel('success Rate', color='g')
ax2.plot(labels, y2, 'g--')
ax.set_title('Number of Startups per City')
ax.legend()
# fig.show()
fig.savefig('Number_of_Startups_per_City.png', dpi=1000, transparent=True)


# ## 画前15个行业分布的初创公司

# In[57]:


# 数据处理
industry_startups = startups.groupby(['market']).agg({'name': 'nunique',
                                                    'success': "sum"}).sort_values(by="name",ascending=False)

industry_startups['fail'] = industry_startups['name'] - industry_startups['success']
industry_startups['success_per'] = industry_startups['success'] / industry_startups['name']

industry_startups=industry_startups.head(10)
industry_startups.reset_index(inplace=True)
industry_startups


# In[58]:


# 画图
labels = industry_startups.market
success_means = industry_startups.success
fail_means = industry_startups.fail
y2=industry_startups.success_per


width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.bar(labels,success_means, width, label='success')
ax.bar(labels, fail_means, width, bottom=success_means,
       label='Fail')
ax.set_xlabel('Market')
ax.set_ylabel('Startups Num')


ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right") # ha表示文字的哪个位置是对应坐标轴的位置参考

ax2.set_ylabel('success Rate', color='g')
ax2.plot(labels, y2, 'g--')
ax.set_title('Number of Startups per Market')
ax.legend()
# fig.show()
fig.savefig('Number_of_Startups_per_Market.png', dpi=1000, transparent=True)


# ## 画教育信息的图

# In[59]:


startups.head()


# In[60]:


# 数据处理
edu_info = startups.groupby(['success']).agg({'BS': 'sum','BA': 'sum','MBA': 'sum','MS': 'sum','PhD': 'sum','unknown': 'sum'})

edu_info.reset_index(inplace=True)
edu_info


# In[61]:


# 画图(有unknown)
labels = ['fail','success']
BS_means = edu_info.BS
BA_means = edu_info.BA
MS_means = edu_info.MS
MBA_means = edu_info.MBA
PhD_means = edu_info.PhD
unknown_means = edu_info.unknown


width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots(figsize=(8,12))

ax.bar(labels,BS_means, width, label='BS')
ax.bar(labels, BA_means, width, bottom=BS_means,label='BA')
ax.bar(labels, MS_means, width, bottom=BS_means+BA_means,label='MS')
ax.bar(labels, MBA_means, width, bottom=BS_means+BA_means+MS_means,label='MBA')
ax.bar(labels, PhD_means, width, bottom=BS_means+BA_means+MS_means+MBA_means,label='PhD')
ax.bar(labels, unknown_means, width, bottom=BS_means+BA_means+MS_means+MBA_means+PhD_means,label='unknown')


ax.set_xlabel('Status of Startups')
ax.set_ylabel('Num of Entrepreneurs with Different Education Levels')
ax.set_title('Relationships between Education Levels and success')
ax.legend()
# fig.show()
fig.savefig('Relationships_between_Education_Levels_and_success(with unknown).png', dpi=1000, transparent=True)


# In[62]:


# 画图(无unknown)
labels = ['fail','success']
BS_means = edu_info.BS
BA_means = edu_info.BA
MS_means = edu_info.MS
MBA_means = edu_info.MBA
PhD_means = edu_info.PhD
unknown_means = edu_info.unknown


width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots(figsize=(8,12))

ax.bar(labels,BS_means, width, label='BS')
ax.bar(labels, BA_means, width, bottom=BS_means,label='BA')
ax.bar(labels, MS_means, width, bottom=BS_means+BA_means,label='MS')
ax.bar(labels, MBA_means, width, bottom=BS_means+BA_means+MS_means,label='MBA')
ax.bar(labels, PhD_means, width, bottom=BS_means+BA_means+MS_means+MBA_means,label='PhD')


ax.set_xlabel('Status of Startups')
ax.set_ylabel('Num of Entrepreneurs with Different Education Levels')


ax.set_title('Relationships between Education Levels and success')
ax.legend()
# fig.show()
fig.savefig('Relationships_between_Education_Levels_and_success(without unknown).png', dpi=1000, transparent=True)


# ## 画每年的初创公司密度分布

# In[63]:


plt.figure(figsize=(16,8))
sns.kdeplot(startups['founded_year'], color="black", legend=False)
plt.xlabel('founded year',fontsize=14)
plt.ylabel('density',fontsize=14)
plt.tight_layout()
plt.title('Proportion of companies by founding year')
plt.savefig('year_of_founding.png', dpi=300, transparent=True)


# # 为了正确选择特征处理数据

# In[64]:


# 选一些参数参与训练，并且将含有缺失值的数据行删除，第二次数据锐减，这里可以调整
startups_new=startups.drop(['id','name','status','public_at','acquired_at'], axis=1)
startups_new=startups_new.dropna(axis=0,subset = ['market', 
                   'country_code',
                   'city',
                   'funding_rounds', 
                   'funding_total_usd',
                    'relationships',
                   'founded_year', 
                   'numEntrepreneurs',
                   'relationships'
                          ])


# In[65]:


# 看一眼剩下的数据规模和成功企业比例
startups_new.success.value_counts()


# In[66]:


# 提取标签
st_status = startups_new[['success']]


# In[67]:


# 提取特征
st_features = startups_new.drop('success',axis=1)


# In[68]:


# 利用pandas的方法实现one hot encode
st_features = pd.get_dummies(st_features)


# In[69]:


# 把label和特征合并在一起
st_master = pd.concat([st_status,st_features],axis=1)


# In[70]:


# 初步设置训练集
# 后面的列是特征
X_log_success = st_master.drop('success',axis=1)
# 第一列是label
y_log_success = st_master['success']


# In[71]:


# 特征提取，从我们手动添加的特征中提取5个相关程度最高的特征
selector_f = SelectKBest(f_regression, k='all')
selector_f.fit(X_log_success, y_log_success)

name_var = [X_log_success.columns]
var_score = selector_f.scores_

kbest = pd.DataFrame(var_score, name_var).sort_values(by=0, ascending=False).head(20).reset_index()
kbest.columns = ['var', 'score']


# In[72]:


f_reg_var = kbest['var'].tolist()


# In[73]:


# 前几个相关性高的特征
f_reg_var


# # 根据相关性选用特征重新进行数据处理

# In[74]:


startups_new.head()


# In[75]:


startups_new=startups.drop(['id','name','status','public_at','acquired_at','funding_rounds','funding_total_usd','relationships'], axis=1)
startups_new=startups_new.dropna(axis=0,subset = ['market', 
                   'country_code',
                   'city',
                   'founded_year', 
                   'numEntrepreneurs',
                          ])


# In[76]:


# 提取标签
st_status = startups_new[['success']]


# In[77]:


# 提取特征
st_features = startups_new.drop('success',axis=1)


# In[78]:


# 利用pandas的方法实现one hot encode
st_features = pd.get_dummies(st_features)


# In[79]:


# 把label和特征合并在一起
st_master = pd.concat([st_status,st_features],axis=1)


# In[80]:


# 数据均衡，调整正反例数据比例，对多的数据进行下采样，用于避免机器学习时产生误差

# 区分哪个标签的数量多
if st_master.success.mean()<0.5:
    success_majority = st_master[st_master.success==0]
    success_minority = st_master[st_master.success==1]
else:
    success_majority = st_master[st_master.success==1]
    success_minority = st_master[st_master.success==0]
    
# 对数量多的标签数据进行下采样
success_majority_downsampled = resample(success_majority, 
                                 replace=False,    # 采样后不替换原数据
                                 n_samples=success_minority.success.count(),
                                 random_state=123) # 固定随机种子确保每次运行实验结果都相同
 
# 把降采样后的标签与小规模标签数据融合
success_downsampled = pd.concat([success_majority_downsampled, success_minority])


# In[81]:


# 看一眼最终的数据集大小
success_downsampled.success.value_counts()


# In[82]:


# 初步设置训练集
# 后面的列是特征
X_log_success = success_downsampled.drop('success',axis=1)
# 第一列是label
y_log_success = success_downsampled['success']


# In[83]:


# 特征提取，从我们手动添加的特征中提取5个相关程度最高的特征
selector_f = SelectKBest(f_regression, k='all')
selector_f.fit(X_log_success, y_log_success)

name_var = [X_log_success.columns]
var_score = selector_f.scores_

kbest = pd.DataFrame(var_score, name_var).sort_values(by=0, ascending=False).head(20).reset_index()
kbest.columns = ['var', 'score']


# In[84]:


f_reg_var = kbest['var'].tolist()


# In[85]:


# # 最终采用的特征
f_reg_var


# # 训练用数据集准备

# In[86]:


success_log = sm.Logit(success_downsampled['success'], success_downsampled[f_reg_var]).fit()
print(success_log.summary())


# In[87]:


master_success = sklearn.utils.resample(success_downsampled, random_state=3) #  此处的随机数对结果影响较大

# 重新确定训练特征和label
X_success = master_success[f_reg_var]

y_success = master_success['success']

# 分训练集和测试集
X_success_train, X_success_test, y_success_train, y_success_test = train_test_split(X_success, 
                                                                                        y_success,
                                                                                        test_size=0.3,
                                                                                        random_state=5)

print(X_success_train.shape, y_success_train.shape)
print(X_success_test.shape, y_success_test.shape)


# # 训练模型

# In[88]:


# 逻辑回归训练
logit = LogisticRegression()
model_logit = logit.fit(X_success_train, y_success_train)


# In[89]:


#随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier (random_state = 42)
model_rf = rf.fit(X_success_train, y_success_train)


# In[90]:


#SVM
from sklearn import svm
model_svm = svm.SVC(C=2, kernel='linear', gamma=8)
model_svm = model_svm.fit(X_success_train, y_success_train)


# In[91]:


#XGBoost
import xgboost as xgb

xgb_classifier = xgb.XGBClassifier()
model_xgb = xgb_classifier.fit(X_success_train, y_success_train)


# # 进行预测

# In[92]:


# 逻辑回归预测
prediction_logit = model_logit.predict(X_success_test)
print("Score:", model_logit.score(X_success_test, y_success_test))


# In[93]:


#随机森林预测
predictions_rf = model_rf.predict(X_success_test)
print("Score:", model_rf.score(X_success_test, y_success_test))


# In[94]:


#SVM预测
predictions_svm = model_svm.predict(X_success_test)
print("Score:", model_svm.score(X_success_test, y_success_test))


# In[95]:


#XGBoost 预测
predictions_xgb = model_xgb.predict(X_success_test)
print("Score:", model_xgb.score(X_success_test, y_success_test))


# # 使用指标衡量预测结果好坏

# In[96]:


# 其他衡量指标
kfold = model_selection.KFold(n_splits=10, random_state=5)
results_logit_acc = model_selection.cross_val_score(logit, X_success_train, y_success_train, cv=kfold, scoring='accuracy')
results_logit_pre = model_selection.cross_val_score(logit, X_success_train, y_success_train, cv=kfold, scoring='precision')
results_logit_rec = model_selection.cross_val_score(logit, X_success_train, y_success_train, cv=kfold, scoring='recall')

print("10-fold cross validation average accuracy: %.3f" % (results_logit_acc.mean()))
print("10-fold cross validation average precision: %.3f" % (results_logit_pre.mean()))
print("10-fold cross validation average recall: %.3f" % (results_logit_rec.mean()))


# # 画混淆矩阵

# In[97]:


#绘制混淆矩阵函数
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[108]:


# 逻辑回归混淆矩阵
confusion_matrix_log = confusion_matrix(y_success_test, prediction_logit)
class_names=["fail","success"]    

# Plot non-normalized confusion matrix
#plt.figure(figsize=(8,8))
#plot_confusion_matrix(confusion_matrix_log, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(4,4))
plot_confusion_matrix(confusion_matrix_log, classes=class_names, normalize=True)
plt.savefig('confusion_matrix_logistic_regression.png', dpi=300, transparent=True)
plt.show()


# In[109]:


# 随机森林混淆矩阵
confusion_matrix_log_rf = confusion_matrix(y_success_test, predictions_rf)
class_names_rf=["fail","success"]    

# Plot non-normalized confusion matrix
#plt.figure(figsize=(8,8))
#plot_confusion_matrix(confusion_matrix_log, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(4,4))
plot_confusion_matrix(confusion_matrix_log_rf, classes=class_names_rf, normalize=True)
plt.savefig('confusion_matrix_rf.png', dpi=300, transparent=True)
plt.show()


# In[110]:


# SVM混淆矩阵
confusion_matrix_log_svm = confusion_matrix(y_success_test, predictions_svm)
class_names_svm=["fail","success"]    

# Plot non-normalized confusion matrix
#plt.figure(figsize=(8,8))
#plot_confusion_matrix(confusion_matrix_log, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(4,4))
plot_confusion_matrix(confusion_matrix_log_svm, classes=class_names_svm, normalize=True)
plt.savefig('confusion_matrix_svm.png', dpi=300, transparent=True)
plt.show()


# In[111]:


# XGBoost混淆矩阵
confusion_matrix_log_xgb = confusion_matrix(y_success_test, predictions_xgb)
class_names_xgb=["fail","success"]    

# Plot non-normalized confusion matrix
#plt.figure(figsize=(8,8))
#plot_confusion_matrix(confusion_matrix_log, classes=class_names,
#                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(figsize=(4,4))
plot_confusion_matrix(confusion_matrix_log_xgb, classes=class_names_xgb, normalize=True)
plt.savefig('confusion_matrix_xgb.png', dpi=300, transparent=True)
plt.show()


# In[ ]:




