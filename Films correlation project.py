#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

#read de df
df = pd.read_csv(r'C:\Users\juani\Downloads\movies.csv\movies.csv')


# In[3]:


df.head(10)


# In[3]:


# missing data? 

for col in df.columns:
    percent_missing = np.mean(df[col].isnull())
    print('{} - {}%' .format(col, percent_missing))
    df = df.dropna()


# In[4]:


# type of data for columns 
df.dtypes


# In[11]:


#chage some dtypes
df['budget'] = df['budget'].astype('float64')

df['votes'] = df['budget'].astype('float64')

df['gross'] = df['gross'].astype('float64')


# In[12]:


df.dtypes


# In[13]:


#correct the year 
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)
df


# In[14]:


##
df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[15]:


#duplicates?
df.drop_duplicates()


# In[16]:


#scatter budget vs gross
plt.scatter(x=df['budget'], y=df['gross'])
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross earnings')
plt.ylabel('Budget for film')
plt.show()


# In[17]:


df.head(10)


# In[44]:



# budget vs gross plot
sns.regplot(x='gross', y='budget', data=df, scatter_kws={"color": "red"}, line_kws={"color": "blue"})


# In[23]:


#looking for correlation
df.corr(method = 'pearson')


# In[29]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matric for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')

plt.show()


# In[27]:


#looking for correlation
df.corr(method = 'kendall')


# In[ ]:





# In[ ]:





# In[34]:


correlation_matrix = df.corr(method='pearson')
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matric for numeric features')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[ ]:





# In[40]:


##company
df_numerized = df.copy()

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
df_numerized  


# In[41]:


df_numerized.corr()


# In[49]:


correlation_matrix = df_numerized.corr()
corr_pairs = correlation_matrix.unstack()
corr_pairs


# In[46]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs 


# In[50]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




