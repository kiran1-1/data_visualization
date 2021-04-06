#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd #pandas is a lib for data manipulation and data readinf(reading csv file)
#import pandas_datareader as pdr
import numpy as np #arranging data in array form and doing operations
import matplotlib.pyplot as plt   #for datavisualization(graph plotting)
import seaborn as sns           #for datavisualization,3d format(heatmap)
#from pandas import DataFrame


# In[4]:


data=pd.read_csv(r"C:\Users\Lenovo\Downloads\covid19.csv")
data


# # Pie Chart

# In[5]:


dataframe=data.head(20)   #to choose first 20 data


# In[10]:



fig = plt.figure(dpi = 128, figsize = (10,10)) #dpi-dots per inches (to fit in canva),to show canva of fig(10 by 10)
label=dataframe['state_ut']                 
sizes=dataframe['confirmed_cases']
plt.pie(sizes,labels=label,autopct='%1.2f%%')    #autopct-to show data in percentafe
#plt.axis('equal')
plt.title('piechart',loc='right',fontsize=50)
plt.show()


# # Bar Plot

# In[16]:


fig = plt.figure(dpi = 128, figsize = (10,10))
#df=data.tail(5)
x=data['state_ut']
y=data['death_cases']
plt.xlabel('states')   #its for labelling
plt.ylabel('deathcases')
ax = fig.add_axes([0,0,1,1])   #its for showing rectangular axes
plt.title('Barplot',loc='center',fontsize=50)
ax.barh(x,y)   #barh-bar horizontally
plt.show()


# # Histogram

# In[123]:


#fig=plt.figure(dpi=128,figsize=[10,10])
ax = plt.subplots(1,1)
a = data['death_cases']
label=data['state_ut']
#plt.label('activecases')
plt.ylabel('frequency')
plt.xticks( a,label, rotation ='vertical') 
#b=[min(data['state_ut']),max(data['state_ut'])]
plt.hist(a,bins=10)   
plt.title('Histogram',loc='center',fontsize=50)
plt.show()
b


# # dotted graph

# In[69]:


fig = plt.figure(dpi = 128, figsize = (8,8))
x=data['state_ut']
y=data['active_cases']
z=data['death_cases']
plt.plot(y,x,linestyle='dotted')
plt.plot(z,x,linestyle='dashed')#linstyle is for type of styling it can be dashed,dotted,line etc
plt.ylabel('active cases')
plt.title('Dotted graph',loc='center',fontsize=50)
plt.show()


# # line plot

# In[72]:


fig = plt.figure( figsize = (10,10))
y=data['state_ut']
x=data['active_cases']
a=data['death_cases']
plt.xlabel('activecases')
plt.plot(x,y)
plt.plot(a,y)
plt.title('lineplot',loc='center',fontsize=50)
plt.show()


# # Grid  graph

# In[78]:


fig, axes = plt.subplots(1,3, figsize = (10,10))
x=data['state_ut']
y=data['active_cases']
z=data['confirmed_cases']
d=data['death_cases']
plt.title('Grid',loc='left',fontsize=30)
axes[0].plot(y, x, 'g',lw=2)
axes[0].grid(True)
axes[0].set_title('activecases grid')
axes[1].plot(z,x, 'r')   #r is for ccolor
axes[1].grid(color='b', ls = '-.', lw = 0.25)
axes[1].set_title('confirmed grid')
axes[2].plot(d,x)
axes[2].grid(True)
axes[2].set_title('deathcases grid')
fig.tight_layout()
#plt.title('Grid',loc='left',fontsize=30)
plt.show()


# # scatter plot

# In[20]:


fig = plt.figure(dpi = 128, figsize = (10,10))
x=data['state_ut']
y=data['active_cases']
z=data['recovered_cases']
plt.scatter(y,x)
plt.scatter(z,x)
plt.title('Scatterplot',loc='center',fontsize=50)
plt.show()


# In[86]:


fig=plt.figure(dpi=128,figsize=(10,10))
x=data['state_ut']
y=data['recovered_cases']
z=data['confirmed_cases']
plt.plot(y,x,'o:r',linestyle='dashed') 
plt.plot(z,x,'*:g',linestyle='dashed') #o:r-o is for points it can be star ,dash and r is for red
plt.ylabel('recovered and confirmed cases')
plt.title('Marker plot')
plt.show()


# # stack plot

# In[87]:


fig = plt.figure(dpi = 128, figsize = (10,10))
x=data['death_cases']
y=data['active_cases']
plt.bar(range(len(x)),x)
plt.bar(range(len(y)),y,bottom=x)
plt.xlabel('deadthcases')
plt.ylabel('activecases')
plt.title('Stackplot')
plt.show()


# In[88]:


fig = plt.figure(dpi = 128, figsize = (10,10))
x=data['state_ut']
y=data['active_cases']
z=data['death_cases']
plt.bar(np.arange(len(y)),y,width=0.3)
plt.bar(np.arange(len(z)),z)
plt.title('Multistack')
plt.show()


# # radar graph

# In[24]:


fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar')) #projection polar means equatorcircular form
x=data['active_cases']
y=data['death_cases']
z=data['recovered_cases']
ax1.plot(x, y)
ax2.plot(x, z)
plt.title('radar graph')
plt.show()


# # step plot

# In[89]:


fig = plt.figure(dpi = 128, figsize = (10,10))
x=data['state_ut']
y=data['death_cases']
plt.xlabel('death_cases')
plt.step(y,x)
plt.title('Stepplot')
plt.show()


# # box plot

# In[90]:


fig = plt.figure(dpi = 128, figsize = (10,10))
df=data[['state_ut','active_cases','death_cases']]
x=df['state_ut']
y=df['death_cases']
ax=sns.boxplot(x=y,y=x,data=df)
plt.title('boxplot')
plt.show()


# In[91]:


fig = plt.figure(dpi = 128, figsize = (10,10))
xlist = data['active_cases']
ylist = data['death_cases']
X, Y = np.meshgrid(xlist, ylist)
Z = np.sqrt(X**2 + Y**2)
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('activedeath cases')
plt.title('contourplot')
plt.show()


# # Violoin plot

# In[102]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
x=data['confirmed_cases']
y=data['recovered_cases']
z=data['death_cases']
v=data['active_cases']
ax.set_title('Violinplot')
df=[x,y,z,v]
ax.violinplot(df)
plt.show()


# # bubble chart

# In[104]:


import plotly.graph_objects as go
#df=data.tail(20)
x=data['active_cases']
y=data['death_cases']
fig = go.Figure(data=[go.Scatter(
    x=x, y=y,
    mode='markers',
    marker_size=[40, 60, 80, 100])
])
go.Figure(layout_title_text="BUBBLECHART")
fig.show()


# # Waterfall

# In[3]:


pip install waterfallcharts


# In[100]:


import waterfall_chart
get_ipython().run_line_magic('matplotlib', 'inline')
a=data['active_cases']
b=data['death_cases']
waterfall_chart.plot(a, b)


# # Treemap

# In[39]:


pip install squarify


# In[105]:


import squarify
x=len(data['active_cases'])
y=len(data['death_cases'])
z=len(data['recovered_cases'])
e=len(data['confirmed_cases'])
label=['active','death','recovered','confirmed']
size=[x,y,z,e]
squarify.plot(size,label=label)
plt.title('Treemap')
plt.show()


# # Area chart

# In[106]:


fig = plt.figure(dpi = 128, figsize = (10,10))
x=data['state_ut']
y=data['active_cases']
plt.fill_between(y, x)
plt.title('Areachart')
plt.show()


# # heatmap

# In[114]:


fig = plt.figure(dpi = 128, figsize = (10,10))
a=data['active_cases']
b=data['recovered_cases']
c=data['confirmed_cases']
d=data['death_cases']
plt.title('Heatmap')
dat=[a,c,b,d]
p1 = sns.heatmap(dat)


# In[29]:


from statsmodels.graphics.mosaicplot import mosaic
#fig = plt.figure(dpi = 128, figsize = (20,20))
fig = plt.figure( figsize = (10,10))
d=data.head(5)
active_cases= d['active_cases']
state_ut=d['state_ut']
data = pd.DataFrame({'active_cases': active_cases, 'state_ut': state_ut})
mosaic(data, ['active_cases', 'state_ut'])
plt.title('mosaic plot',fontsize=50)
plt.show()


# In[ ]:





# In[ ]:




