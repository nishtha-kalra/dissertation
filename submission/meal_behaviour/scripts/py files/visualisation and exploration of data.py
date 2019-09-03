
# coding: utf-8

# In[1]:


# loading required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_sas('DR1IFF_I.XPT')


# ### Data cleaning

# In[2]:


data = data.dropna()
data = data.sort_values(by=['DR1_020'])
data['DR1_020'] = data['DR1_020']/3600
data.drop(data[data.DR1ICARB < 5e-20].index, inplace=True)
data.drop(data[data.DR1_020 < 1e-20].index, inplace=True)


# ### Time vs CHO

# In[3]:


X = data[data.SEQN == 83732]
X = X[['SEQN', 'DR1ICARB', 'DR1_020']].copy()


# In[4]:


plt.plot(X.DR1_020, X.DR1ICARB)
plt.xlabel('Time in hours')
plt.ylabel('CHO in gm')
plt.savefig('timevscho.png')


# ### Relation between CHO and days of week

# In[5]:


data = data.dropna()
X = data[['SEQN', 'DR1_020', 'DR1DAY', 'DR1ICARB']].copy()


# In[6]:


sun = 0
mon = 0
tue = 0
wed = 0
thu = 0
fri = 0
sat = 0
sunc = 0
monc = 0
tuec = 0
wedc = 0
thuc = 0
fric = 0
satc = 0


# In[7]:


for row, value in X.iterrows():
    if value['DR1DAY'] == 1.0:
        sunc += 1
        sun = sun + value['DR1ICARB']
    if value['DR1DAY'] == 2.0:
        monc += 1
        mon = mon + value['DR1ICARB']
    if value['DR1DAY'] == 3.0:
        tuec += 1
        tue = tue + value['DR1ICARB']
    if value['DR1DAY'] == 4.0:
        wedc += 1
        wed = wed + value['DR1ICARB']
    if value['DR1DAY'] == 5.0:
        thuc += 1
        thu = thu + value['DR1ICARB']
    if value['DR1DAY'] == 6.0:
        fric += 1
        fri = fri + value['DR1ICARB']
    if value['DR1DAY'] == 7.0:
        satc += 1
        sat = sat + value['DR1ICARB']


# In[8]:


week1 = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']
week = [sun/sunc, mon/monc, tue/tuec, wed/wedc, thu/thuc, fri/fric, sat/satc]


# In[9]:


plt.bar(np.arange(len(week1)), week)
plt.xticks(np.arange(len(week1)), week1)
plt.ylabel('average CHO level in gm')
plt.show
plt.savefig('dayvscho.png')


# People generally take high CHO diet on Sunday, Friday and Saturday.

# ### Relation between CHO and Language respondent used mostly

# In[10]:


lang = data[['SEQN', 'DR1_020', 'DR1LANG', 'DR1ICARB']].copy()


# In[11]:


English = 0
Spanish = 0
EnglishandSpanish = 0
Other = 0
Asian = 0
AsianandEnglish = 0
Englishc = 0
Spanishc = 0
EnglishandSpanishc = 0
Otherc = 0
Asianc = 0
AsianandEnglishc = 0


# In[12]:


for row, value in lang.iterrows():
    if value['DR1LANG'] == 1.0:
        Englishc += 1
        English = English + value['DR1ICARB']
    if value['DR1LANG'] == 2.0:
        Spanishc += 1
        Spanish = Spanish + value['DR1ICARB']
    if value['DR1LANG'] == 3.0:
        EnglishandSpanishc += 1
        EnglishandSpanish = EnglishandSpanish + value['DR1ICARB']
    if value['DR1LANG'] == 4.0:
        Otherc += 1
        Other = Other + value['DR1ICARB']
    if value['DR1LANG'] == 5.0:
        Asianc += 1
        Asian = Asian + value['DR1ICARB']
    if value['DR1LANG'] == 6.0:
        AsianandEnglishc += 1
        AsianandEnglish = AsianandEnglish + value['DR1ICARB']


# In[13]:


langc = ['English', 'Spanish', 'English\n and Spanish', 'Other', 'Asian', 'Asian\n and English']
lang = [English/Englishc, Spanish/Spanishc, EnglishandSpanish/EnglishandSpanishc, Other/Otherc, Asian/Asianc, AsianandEnglish/AsianandEnglishc]


# In[14]:


plt.bar(np.arange(len(langc)), lang)
plt.xticks(np.arange(len(langc)), langc)
plt.ylabel('average CHO level in gm')
plt.savefig('langvscho.png')


# People speaking English and Spanish consume most Carbohydrates while people sepaking Asian languages consume least Carbohydrates

# ### Relation between CHO level and Combination food type 

# In[15]:


food = data[['SEQN', 'DR1_020', 'DR1CCMTX', 'DR1ICARB']].copy()


# In[16]:


fooddict = {
    1 : { 'name' : 'beverage', 'count' : 0, 'cho' : 0},
    2 : { 'name' : 'cereal', 'count' : 0, 'cho' : 0},
    3 : { 'name' : 'bread', 'count' : 0, 'cho' : 0},
    4 : { 'name' : 'salad', 'count' : 0, 'cho' : 0},
    5 : { 'name' : 'sandwich', 'count' : 0, 'cho' : 0}, 
    6 : { 'name' : 'soup', 'count' : 0, 'cho' : 0},
    7 : { 'name' : 'frozen', 'count' : 0, 'cho' : 0},
    8 : { 'name' : 'icecream', 'count' : 0, 'cho' : 0},
    9 : { 'name' : 'driedbean', 'count' : 0, 'cho' : 0},
    10 : { 'name' : 'fruit', 'count' : 0, 'cho' : 0},
    11 : { 'name' : 'tortilla', 'count' : 0, 'cho' : 0},
    12 : { 'name' : 'meat', 'count' : 0, 'cho' : 0},
    13 : { 'name' : 'luncables', 'count' : 0, 'cho' : 0},
    14 : { 'name' : 'chips', 'count' : 0, 'cho' : 0}
}


# In[17]:


for key in fooddict.keys():
    for row, value in food.iterrows():
        if value['DR1CCMTX'] == key:
            fooddict[key]['count'] += 1
            fooddict[key]['cho'] += value['DR1ICARB']


# In[18]:


foodnames = []
foodcho = []
for key in fooddict:
    foodnames.append(fooddict[key]['name'])
    foodcho.append(fooddict[key]['cho']/fooddict[key]['count'])


# In[19]:


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
plt.bar(np.arange(len(foodnames)), foodcho)
plt.xticks(np.arange(len(foodnames)), foodnames)
plt.ylabel('average CHO level in gm')
plt.savefig('foodvscho.png')


# ### Relation of CHO with DR2_030Z - Name of eating occasion

# In[20]:


occasion = data[['DR1_030Z', 'DR1ICARB']].copy()


# In[21]:


occasiondict = {
    1 : { 'name' : 'Breakfast\nDesayano', 'count' : 0, 'cho' : 0},
    2 : { 'name' : 'Lunch\nAlmuerzo\nCena', 'count' : 0, 'cho' : 0},
    3 : { 'name' : 'Dinner', 'count' : 0, 'cho' : 0},
    6 : { 'name' : 'Snack\nMerienda\nBotana\nBocadillo\nTentempie', 'count' : 0, 'cho' : 0},
    7 : { 'name' : 'Drink\nBebida', 'count' : 0, 'cho' : 0},
}


# Merging keys of spanish words with english words

# In[22]:


for row, value in data.iterrows():
    if value['DR1_030Z'] == 10:
        value['DR1_030Z'] = 1
    elif value['DR1_030Z'] == 11 or value['DR1_030Z'] == 12 or value['DR1_030Z'] == 5:
        value['DR1_030Z'] = 2
    elif value['DR1_030Z'] == 14:
        value['DR1_030Z'] = 3
    elif value['DR1_030Z'] == 4 or value['DR1_030Z'] == 9 or value['DR1_030Z'] == 13 or value['DR1_030Z'] == 15 or value['DR1_030Z'] == 16 or value['DR1_030Z'] == 17 or value['DR1_030Z'] == 18:
        value['DR1_030Z'] = 6
    elif value['DR1_030Z'] == 19:
        value['DR1_030Z'] = 7


# Calculating count and carbohydrate level of each eating occasion

# In[23]:


for key in occasiondict.keys():
    for row, value in occasion.iterrows():
        if value['DR1_030Z'] == key:
            occasiondict[key]['count'] += 1
            occasiondict[key]['cho'] += value['DR1ICARB']


# Creating list of names of occasion of food and average CHO level of each food ocassion

# In[24]:


occasionnames = []
occasioncho = []
for key in occasiondict: 
    occasionnames.append(occasiondict[key]['name'])
    occasioncho.append(occasiondict[key]['cho']/occasiondict[key]['count'])


# In[25]:


fig_size[0] = 10
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.bar(np.arange(len(occasionnames)), occasioncho)
plt.xticks(np.arange(len(occasionnames)), occasionnames)
plt.ylabel('average CHO level in gm')
plt.savefig('occasionvscho.png')


# ### Relation between CHO levels and DR2_040Z - Did you eat this meal at home?

# In[26]:


athome = data[['DR1_040Z', 'DR1ICARB']].copy()


# In[27]:


yes = 0
no = 0
yesc = 0
noc = 0
for row, value in athome.iterrows():
    if value['DR1_040Z'] == 1.0:
        yesc += 1
        yes = yes + value['DR1ICARB']
    if value['DR1_040Z'] == 2.0:
        noc += 1
        no = no + value['DR1ICARB']


# In[28]:


plt.bar([0,1], [yes/yesc,no/noc])
plt.xticks(np.arange(2), ['at home', 'not at home'])
plt.ylabel("Average CHO levels in gm")
plt.savefig('homevscho.png')


# Food not cosumed at home has more carbohydrates
