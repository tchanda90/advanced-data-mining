
# Clustream


```python
import pandas as pd
import math
import numpy as np
from collections import Counter
from scipy.spatial import distance
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
import warnings
warnings.filterwarnings('ignore')

file = 'C:\\My Files\\OVGU\\Hiwi\\data mining 2\\advanced-data-mining\\data\\data.csv'
```

### Data points from T1 to T11 are the initial points, so we will apply K-Means on this initial set of points


```python
full_data = pd.read_csv(file).set_index('T')
initial = full_data[0:11].copy()
online = full_data[11:].copy()
initial
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>T</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.5</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



### The following points are used as the initial centroids


```python
d = {'Centers': ['C1', 'C2', 'C3', 'C4', 'C5'], 'X': [1, 2.5, 2, 4, 6], 'Y': [1, 2, 7, 7, 2]}
centroids = pd.DataFrame(d).set_index('Centers')
centroids
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>Centers</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C1</th>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>C2</th>
      <td>2.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>C3</th>
      <td>2.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>C4</th>
      <td>4.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>C5</th>
      <td>6.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



### Now we will apply K-Means on the initial data points, and assign every point to its nearest centroid.
### After K-Means converges, the final centroids and the assignment of each point are plotted below


```python
kmeans = KMeans(n_clusters=5, random_state=0, init=centroids).fit(initial[['X', 'Y']])
labels = kmeans.labels_
# Change the cluster labels from just numbers like 0, 1, 2 to MC1, MC2, etc
labels = ['MC'+str((x+1)) for x in labels]
initial['Centroid'] = labels
display(initial)

# Plot the K-means output with assignments
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="X", y="Y", style="Centroid", hue="Centroid", data=initial, s=150)

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>Centroid</th>
    </tr>
    <tr>
      <th>T</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>2.0</td>
      <td>MC5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>MC5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.5</td>
      <td>1.0</td>
      <td>MC5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>MC1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>MC2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>MC2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3.0</td>
      <td>2.5</td>
      <td>MC2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>8.0</td>
      <td>MC3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.0</td>
      <td>6.0</td>
      <td>MC3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2.5</td>
      <td>7.0</td>
      <td>MC3</td>
    </tr>
    <tr>
      <th>11</th>
      <td>4.0</td>
      <td>7.0</td>
      <td>MC4</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_7_1.png)



```python
def add_to_mc(microcluster, x, y, t):
    # Increment LS for X
    cft_combined.at[microcluster, 'CF1(X)'] += x
    # Increment SS for X
    cft_combined.at[microcluster, 'CF2(X)'] += x**2
    # Increment LS for Y
    cft_combined.at[microcluster, 'CF1(Y)'] += y
    # Increment SS for Yhttp://localhost:8888/notebooks/clustream.ipynb#New-point-at-T12:-(2,-7)
    cft_combined.at[microcluster, 'CF2(Y)'] += y**2
    # Increment LS for T
    cft_combined.at[microcluster, 'CF1(T)'] += t
    # Increment SS for T
    cft_combined.at[microcluster, 'CF2(T)'] += t**2
    # Increment N
    cft_combined.at[microcluster, 'N'] += 1

def create_new_mc(x, y, t, mc_name):

    d = {'CF1(X)': x, 'CF2(X)': x**2, 'CF1(Y)': y, 'CF2(Y)': y**2, 'CF1(T)': t, 'CF2(T)': t**2, 'N': 1}
    row = pd.Series(d, name=mc_name)
    df = cft_combined.append(row)
    df.fillna(0, inplace=True)
    return df

def delete_oldest_mc():
    # idxmin() returns the index of the oldest value of the column
    oldest_mc = cft_combined['Mean(T)'].idxmin()
    return cft_combined.drop(oldest_mc)

def merge_mc(mc1, mc2, new_mc):
    feature_vector = ['CF1(X)', 'CF2(X)', 'CF1(Y)', 'CF2(Y)', 'CF1(T)', 'CF2(T)', 'N']
    for column in feature_vector:
        cft_combined.at[new_mc, column] = cft_combined.at[mc1, column] + cft_combined.at[mc2, column]
    cft_combined.drop([mc1, mc2], inplace=True)
    cft_combined.fillna(0, inplace=True)
    return cft_combined
        

def recalculate_summaries():
        
    for index, row in cft_combined.iterrows():
    
        cft_combined.at[index, 'Center(X)'] = row['CF1(X)']/row['N']
        cft_combined.at[index, 'Center(Y)'] = row['CF1(Y)']/row['N']

        radius_x = math.sqrt( row['CF2(X)']/row['N'] - (row['CF1(X)']/row['N'])**2 )
        radius_y = math.sqrt( row['CF2(Y)']/row['N'] - (row['CF1(Y)']/row['N'])**2 )
        radius = np.mean([radius_x, radius_y])
        cft_combined.at[index, 'Radius(X)'] = radius_x
        cft_combined.at[index, 'Radius(Y)'] = radius_y
        cft_combined.at[index, 'Radius'] = radius

        cft_combined.at[index, 'Mean(T)'] = row['CF1(T)']/row['N']
        cft_combined.at[index, 'Sigma(T)'] = math.sqrt( row['CF2(T)']/row['N'] - (row['CF1(T)']/row['N'])**2 )

        cft_combined.at[index, 'Max Radius'] = radius * 2

```

### After the initial batch K-Means step, comes the online phase, where new data points arrive one by one. For this, we maintain micro-cluster summaries and update them incrementally as new points arrive. A new point p is handled as follows:
##### 1. Compute the distances between p and each of the q maintained micro-cluster centroids
##### 2. For the closest micro-cluster to p, calculate its max boundary
##### 3. If p is within max boundary, add p to the micro-cluster.
##### 4. If not, delete 1 micro-cluster or merge 2 of the closest located micro-clusters, and create a new micro-cluster with p.
### First, we initialize the micro-cluster summary structure


```python
d = {'CF1(X)': [0.0] * 5, 'CF2(X)': [0.0] * 5, 'CF1(Y)': [0.0] * 5, 'CF2(Y)': [0.0] * 5, 'CF1(T)': [0.0] * 5, 
     'CF2(T)': [0.0] * 5, 'N': [0] * 5, 'MicroCluster': ['MC1', 'MC2', 'MC3', 'MC4', 'MC5']}
cft = pd.DataFrame(d).set_index('MicroCluster')

for row in initial.itertuples():
    c = row.Centroid
    
    cft.at[c, 'CF1(X)'] += row.X
    cft.at[c, 'CF2(X)'] += row.X**2
    
    cft.at[c, 'CF1(Y)'] += row.Y
    cft.at[c, 'CF2(Y)'] += row.Y**2
    
    cft.at[c, 'CF1(T)'] += row.Index
    cft.at[c, 'CF2(T)'] += row.Index**2
    
    cft.at[c, 'N'] += 1

    
# Micro-cluster details
d = {'Center(X)': [0.0] * 5, 'Center(Y)': [0.0] * 5, 'Radius(X)': [0.0] * 5, 'Radius(Y)': [0.0] * 5, 'Mean(T)': [0.0] * 5, 
     'Sigma(T)': [0.0] * 5, 'Radius': [0.0] * 5, 'Max Radius': [0.0] * 5, 'MicroCluster': ['MC1', 'MC2', 'MC3', 'MC4', 'MC5']}
mc_details = pd.DataFrame(d).set_index('MicroCluster')

cft_combined = pd.concat([cft, mc_details], axis=1)
recalculate_summaries()
display(cft_combined)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC1</th>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.0</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>6.5</td>
      <td>14.25</td>
      <td>21.0</td>
      <td>149.00</td>
      <td>27.0</td>
      <td>245.0</td>
      <td>3</td>
      <td>2.166667</td>
      <td>7.000000</td>
      <td>0.235702</td>
      <td>0.816497</td>
      <td>9.0</td>
      <td>0.816497</td>
      <td>0.526099</td>
      <td>1.052199</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>4.0</td>
      <td>16.00</td>
      <td>7.0</td>
      <td>49.00</td>
      <td>11.0</td>
      <td>121.0</td>
      <td>1</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC5</th>
      <td>19.5</td>
      <td>127.25</td>
      <td>6.0</td>
      <td>14.00</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>3</td>
      <td>6.500000</td>
      <td>2.000000</td>
      <td>0.408248</td>
      <td>0.816497</td>
      <td>2.0</td>
      <td>0.816497</td>
      <td>0.612372</td>
      <td>1.224745</td>
    </tr>
  </tbody>
</table>
</div>


### In the data structures above, the Micro-Cluster summaries are calculated from the initial points. Now, we can use the summary information to handle new data points

# TIMEPOINT 12


```python
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[12]], s=50)
p1.text(2-0.8, 7+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC3']['Center(X)'], cft_combined.loc['MC3']['Center(Y)']),
                        1.052199, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_13_0.png)


### New point at T12: (2, 7)
### The point falls inside the Max Boundary of the closest Micro-Cluster MC3, so it is absorbed.
### Since the new point is absorbed by MC3, its feature vector  is updated. 
### The feature vectors BEFORE and AFTER adding the new point are displayed


```python
display(cft_combined)

# Add the new point
new_point = list(online.loc[[12]].iloc[0])
add_to_mc('MC3', new_point[0], new_point[1], 12)

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC3'
                              else '' for i in x], axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC1</th>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.0</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>6.5</td>
      <td>14.25</td>
      <td>21.0</td>
      <td>149.00</td>
      <td>27.0</td>
      <td>245.0</td>
      <td>3</td>
      <td>2.166667</td>
      <td>7.000000</td>
      <td>0.235702</td>
      <td>0.816497</td>
      <td>9.0</td>
      <td>0.816497</td>
      <td>0.526099</td>
      <td>1.052199</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>4.0</td>
      <td>16.00</td>
      <td>7.0</td>
      <td>49.00</td>
      <td>11.0</td>
      <td>121.0</td>
      <td>1</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC5</th>
      <td>19.5</td>
      <td>127.25</td>
      <td>6.0</td>
      <td>14.00</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>3</td>
      <td>6.500000</td>
      <td>2.000000</td>
      <td>0.408248</td>
      <td>0.816497</td>
      <td>2.0</td>
      <td>0.816497</td>
      <td>0.612372</td>
      <td>1.224745</td>
    </tr>
  </tbody>
</table>
</div>





<style  type="text/css" >
    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col0 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col1 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col2 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col3 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col4 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col5 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col6 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col7 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col8 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col9 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col10 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col11 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col12 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col13 {
            background:  lightgreen;
        }    #T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col14 {
            background:  lightgreen;
        }</style><table id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC1</th>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col0" class="data row0 col0" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col1" class="data row0 col1" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col2" class="data row0 col2" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col3" class="data row0 col3" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col4" class="data row0 col4" >4</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col5" class="data row0 col5" >16</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col6" class="data row0 col6" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col7" class="data row0 col7" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col8" class="data row0 col8" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col9" class="data row0 col9" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col10" class="data row0 col10" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col11" class="data row0 col11" >4</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col12" class="data row0 col12" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col13" class="data row0 col13" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row0_col14" class="data row0 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC2</th>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col0" class="data row1 col0" >8</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col1" class="data row1 col1" >22</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col2" class="data row1 col2" >5.5</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col3" class="data row1 col3" >11.25</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col4" class="data row1 col4" >18</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col5" class="data row1 col5" >110</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col6" class="data row1 col6" >3</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col7" class="data row1 col7" >2.66667</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col8" class="data row1 col8" >1.83333</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col9" class="data row1 col9" >0.471405</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col10" class="data row1 col10" >0.62361</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col11" class="data row1 col11" >6</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col12" class="data row1 col12" >0.816497</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col13" class="data row1 col13" >0.547507</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row1_col14" class="data row1 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC3</th>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col0" class="data row2 col0" >8.5</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col1" class="data row2 col1" >18.25</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col2" class="data row2 col2" >28</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col3" class="data row2 col3" >198</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col4" class="data row2 col4" >39</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col5" class="data row2 col5" >389</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col6" class="data row2 col6" >4</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col7" class="data row2 col7" >2.125</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col9" class="data row2 col9" >0.216506</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col10" class="data row2 col10" >0.707107</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col11" class="data row2 col11" >9.75</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col12" class="data row2 col12" >1.47902</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col13" class="data row2 col13" >0.461807</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row2_col14" class="data row2 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC4</th>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col0" class="data row3 col0" >4</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col1" class="data row3 col1" >16</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col2" class="data row3 col2" >7</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col3" class="data row3 col3" >49</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col4" class="data row3 col4" >11</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col5" class="data row3 col5" >121</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col7" class="data row3 col7" >4</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col8" class="data row3 col8" >7</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col11" class="data row3 col11" >11</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC5</th>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col0" class="data row4 col0" >19.5</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col1" class="data row4 col1" >127.25</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col2" class="data row4 col2" >6</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col3" class="data row4 col3" >14</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col4" class="data row4 col4" >6</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col5" class="data row4 col5" >14</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col6" class="data row4 col6" >3</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col7" class="data row4 col7" >6.5</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col8" class="data row4 col8" >2</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col9" class="data row4 col9" >0.408248</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col10" class="data row4 col10" >0.816497</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col11" class="data row4 col11" >2</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col12" class="data row4 col12" >0.816497</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col13" class="data row4 col13" >0.612372</td>
                        <td id="T_b3f52340_66ce_11e9_9595_c3aaa83f8744row4_col14" class="data row4 col14" >1.22474</td>
            </tr>
    </tbody></table>



# TIMEPOINT 13


```python
timepoint = 13
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(2.5-0.8, 3+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC2']['Center(X)'], cft_combined.loc['MC2']['Center(Y)']),
                        1, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_17_0.png)


### New point at T13: (2.5, 3)
### The point falls outside the Max Boundary of the closest Micro-Cluster MC2, so a new Micro-Cluster has to be created.
### In order to accommodate the new Micro-Cluster, an old one must be deleted. The oldest one currently is MC5


```python
display(cft_combined)

# Add the new point
new_point = list(online.loc[[13]].iloc[0])
cft_combined = delete_oldest_mc()
cft_combined = create_new_mc(new_point[0], new_point[1], timepoint, 'MC6')

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC6'
                              else '' for i in x], axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC1</th>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>1</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.00</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>8.5</td>
      <td>18.25</td>
      <td>28.0</td>
      <td>198.00</td>
      <td>39.0</td>
      <td>389.0</td>
      <td>4</td>
      <td>2.125000</td>
      <td>7.000000</td>
      <td>0.216506</td>
      <td>0.707107</td>
      <td>9.75</td>
      <td>1.479020</td>
      <td>0.461807</td>
      <td>0.923613</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>4.0</td>
      <td>16.00</td>
      <td>7.0</td>
      <td>49.00</td>
      <td>11.0</td>
      <td>121.0</td>
      <td>1</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC5</th>
      <td>19.5</td>
      <td>127.25</td>
      <td>6.0</td>
      <td>14.00</td>
      <td>6.0</td>
      <td>14.0</td>
      <td>3</td>
      <td>6.500000</td>
      <td>2.000000</td>
      <td>0.408248</td>
      <td>0.816497</td>
      <td>2.00</td>
      <td>0.816497</td>
      <td>0.612372</td>
      <td>1.224745</td>
    </tr>
  </tbody>
</table>
</div>





<style  type="text/css" >
    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col0 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col1 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col2 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col3 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col4 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col5 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col6 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col7 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col8 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col9 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col10 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col11 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col12 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col13 {
            background:  lightgreen;
        }    #T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col14 {
            background:  lightgreen;
        }</style><table id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC1</th>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col0" class="data row0 col0" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col1" class="data row0 col1" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col2" class="data row0 col2" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col3" class="data row0 col3" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col4" class="data row0 col4" >4</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col5" class="data row0 col5" >16</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col6" class="data row0 col6" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col7" class="data row0 col7" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col8" class="data row0 col8" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col9" class="data row0 col9" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col10" class="data row0 col10" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col11" class="data row0 col11" >4</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col12" class="data row0 col12" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col13" class="data row0 col13" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row0_col14" class="data row0 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC2</th>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col0" class="data row1 col0" >8</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col1" class="data row1 col1" >22</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col2" class="data row1 col2" >5.5</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col3" class="data row1 col3" >11.25</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col4" class="data row1 col4" >18</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col5" class="data row1 col5" >110</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col6" class="data row1 col6" >3</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col7" class="data row1 col7" >2.66667</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col8" class="data row1 col8" >1.83333</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col9" class="data row1 col9" >0.471405</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col10" class="data row1 col10" >0.62361</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col11" class="data row1 col11" >6</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col12" class="data row1 col12" >0.816497</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col13" class="data row1 col13" >0.547507</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row1_col14" class="data row1 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC3</th>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col0" class="data row2 col0" >8.5</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col1" class="data row2 col1" >18.25</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col2" class="data row2 col2" >28</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col3" class="data row2 col3" >198</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col4" class="data row2 col4" >39</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col5" class="data row2 col5" >389</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col6" class="data row2 col6" >4</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col7" class="data row2 col7" >2.125</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col9" class="data row2 col9" >0.216506</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col10" class="data row2 col10" >0.707107</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col11" class="data row2 col11" >9.75</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col12" class="data row2 col12" >1.47902</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col13" class="data row2 col13" >0.461807</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row2_col14" class="data row2 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC4</th>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col0" class="data row3 col0" >4</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col1" class="data row3 col1" >16</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col2" class="data row3 col2" >7</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col3" class="data row3 col3" >49</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col4" class="data row3 col4" >11</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col5" class="data row3 col5" >121</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col7" class="data row3 col7" >4</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col8" class="data row3 col8" >7</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col11" class="data row3 col11" >11</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC6</th>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col0" class="data row4 col0" >2.5</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col1" class="data row4 col1" >6.25</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col2" class="data row4 col2" >3</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col3" class="data row4 col3" >9</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col4" class="data row4 col4" >13</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col5" class="data row4 col5" >169</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col6" class="data row4 col6" >1</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col7" class="data row4 col7" >2.5</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col8" class="data row4 col8" >3</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col9" class="data row4 col9" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col10" class="data row4 col10" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col11" class="data row4 col11" >13</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col12" class="data row4 col12" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col13" class="data row4 col13" >0</td>
                        <td id="T_bb3250cc_66ce_11e9_b8e5_c3aaa83f8744row4_col14" class="data row4 col14" >0</td>
            </tr>
    </tbody></table>



# TIMEPOINT 14


```python
timepoint = 14

# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(3.5-0.8, 7+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC4']['Center(X)'], cft_combined.loc['MC4']['Center(Y)']),
                        1, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_21_0.png)


### New point at T14: (3.5, 7)
### The point falls inside the Max Boundary of the closest Micro-Cluster MC4, so it is absorbed.


```python
display(cft_combined)

# Add the new point
new_point = list(online.loc[[timepoint]].iloc[0])
add_to_mc('MC4', new_point[0], new_point[1], timepoint)

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC4'
                              else '' for i in x], axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC1</th>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>4.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3.0</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.00</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>8.5</td>
      <td>18.25</td>
      <td>28.0</td>
      <td>198.00</td>
      <td>39.0</td>
      <td>389.0</td>
      <td>4.0</td>
      <td>2.125000</td>
      <td>7.000000</td>
      <td>0.216506</td>
      <td>0.707107</td>
      <td>9.75</td>
      <td>1.479020</td>
      <td>0.461807</td>
      <td>0.923613</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>4.0</td>
      <td>16.00</td>
      <td>7.0</td>
      <td>49.00</td>
      <td>11.0</td>
      <td>121.0</td>
      <td>1.0</td>
      <td>4.000000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC6</th>
      <td>2.5</td>
      <td>6.25</td>
      <td>3.0</td>
      <td>9.00</td>
      <td>13.0</td>
      <td>169.0</td>
      <td>1.0</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>





<style  type="text/css" >
    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col0 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col1 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col2 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col3 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col4 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col5 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col6 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col7 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col8 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col9 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col10 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col11 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col12 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col13 {
            background:  lightgreen;
        }    #T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col14 {
            background:  lightgreen;
        }</style><table id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC1</th>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col0" class="data row0 col0" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col1" class="data row0 col1" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col2" class="data row0 col2" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col3" class="data row0 col3" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col4" class="data row0 col4" >4</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col5" class="data row0 col5" >16</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col6" class="data row0 col6" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col7" class="data row0 col7" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col8" class="data row0 col8" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col9" class="data row0 col9" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col10" class="data row0 col10" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col11" class="data row0 col11" >4</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col12" class="data row0 col12" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col13" class="data row0 col13" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row0_col14" class="data row0 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC2</th>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col0" class="data row1 col0" >8</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col1" class="data row1 col1" >22</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col2" class="data row1 col2" >5.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col3" class="data row1 col3" >11.25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col4" class="data row1 col4" >18</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col5" class="data row1 col5" >110</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col6" class="data row1 col6" >3</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col7" class="data row1 col7" >2.66667</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col8" class="data row1 col8" >1.83333</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col9" class="data row1 col9" >0.471405</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col10" class="data row1 col10" >0.62361</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col11" class="data row1 col11" >6</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col12" class="data row1 col12" >0.816497</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col13" class="data row1 col13" >0.547507</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row1_col14" class="data row1 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC3</th>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col0" class="data row2 col0" >8.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col1" class="data row2 col1" >18.25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col2" class="data row2 col2" >28</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col3" class="data row2 col3" >198</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col4" class="data row2 col4" >39</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col5" class="data row2 col5" >389</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col6" class="data row2 col6" >4</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col7" class="data row2 col7" >2.125</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col9" class="data row2 col9" >0.216506</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col10" class="data row2 col10" >0.707107</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col11" class="data row2 col11" >9.75</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col12" class="data row2 col12" >1.47902</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col13" class="data row2 col13" >0.461807</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row2_col14" class="data row2 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC4</th>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col0" class="data row3 col0" >7.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col1" class="data row3 col1" >28.25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col2" class="data row3 col2" >14</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col3" class="data row3 col3" >98</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col4" class="data row3 col4" >25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col5" class="data row3 col5" >317</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col6" class="data row3 col6" >2</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col7" class="data row3 col7" >3.75</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col8" class="data row3 col8" >7</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col9" class="data row3 col9" >0.25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col11" class="data row3 col11" >12.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col12" class="data row3 col12" >1.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col13" class="data row3 col13" >0.125</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row3_col14" class="data row3 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC6</th>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col0" class="data row4 col0" >2.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col1" class="data row4 col1" >6.25</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col2" class="data row4 col2" >3</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col3" class="data row4 col3" >9</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col4" class="data row4 col4" >13</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col5" class="data row4 col5" >169</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col6" class="data row4 col6" >1</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col7" class="data row4 col7" >2.5</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col8" class="data row4 col8" >3</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col9" class="data row4 col9" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col10" class="data row4 col10" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col11" class="data row4 col11" >13</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col12" class="data row4 col12" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col13" class="data row4 col13" >0</td>
                        <td id="T_be5e1512_66ce_11e9_86db_c3aaa83f8744row4_col14" class="data row4 col14" >0</td>
            </tr>
    </tbody></table>



# TIMEPOINT 15


```python
timepoint = 15
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(7-0.8, 8+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC4']['Center(X)'], cft_combined.loc['MC4']['Center(Y)']),
                        1, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_25_0.png)


### New point at T15: (7, 8)
### The point falls outside the Max Boundary of the closest Micro-Cluster MC4, so a new Micro-Cluster has to be created.
### In order to accommodate the new Micro-Cluster, an old one must be deleted. The oldest one currently is MC1


```python
display(cft_combined.copy().style.apply(lambda x: ['background: lightcoral' if x.name == 'MC1'
                                                   else '' for i in x], axis=1))

# Add the new point
new_point = list(online.loc[[timepoint]].iloc[0])
cft_combined = delete_oldest_mc()
cft_combined = create_new_mc(new_point[0], new_point[1], timepoint, 'MC7')

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC7'
                              else '' for i in x], axis=1)
```


<style  type="text/css" >
    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col0 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col1 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col2 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col3 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col4 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col5 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col6 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col7 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col8 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col9 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col10 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col11 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col12 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col13 {
            background:  lightcoral;
        }    #T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col14 {
            background:  lightcoral;
        }</style><table id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC1</th>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col0" class="data row0 col0" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col1" class="data row0 col1" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col2" class="data row0 col2" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col3" class="data row0 col3" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col4" class="data row0 col4" >4</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col5" class="data row0 col5" >16</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col6" class="data row0 col6" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col7" class="data row0 col7" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col8" class="data row0 col8" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col9" class="data row0 col9" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col10" class="data row0 col10" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col11" class="data row0 col11" >4</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col12" class="data row0 col12" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col13" class="data row0 col13" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row0_col14" class="data row0 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC2</th>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col0" class="data row1 col0" >8</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col1" class="data row1 col1" >22</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col2" class="data row1 col2" >5.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col3" class="data row1 col3" >11.25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col4" class="data row1 col4" >18</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col5" class="data row1 col5" >110</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col6" class="data row1 col6" >3</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col7" class="data row1 col7" >2.66667</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col8" class="data row1 col8" >1.83333</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col9" class="data row1 col9" >0.471405</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col10" class="data row1 col10" >0.62361</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col11" class="data row1 col11" >6</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col12" class="data row1 col12" >0.816497</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col13" class="data row1 col13" >0.547507</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row1_col14" class="data row1 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC3</th>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col0" class="data row2 col0" >8.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col1" class="data row2 col1" >18.25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col2" class="data row2 col2" >28</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col3" class="data row2 col3" >198</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col4" class="data row2 col4" >39</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col5" class="data row2 col5" >389</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col6" class="data row2 col6" >4</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col7" class="data row2 col7" >2.125</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col9" class="data row2 col9" >0.216506</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col10" class="data row2 col10" >0.707107</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col11" class="data row2 col11" >9.75</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col12" class="data row2 col12" >1.47902</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col13" class="data row2 col13" >0.461807</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row2_col14" class="data row2 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC4</th>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col0" class="data row3 col0" >7.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col1" class="data row3 col1" >28.25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col2" class="data row3 col2" >14</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col3" class="data row3 col3" >98</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col4" class="data row3 col4" >25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col5" class="data row3 col5" >317</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col6" class="data row3 col6" >2</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col7" class="data row3 col7" >3.75</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col8" class="data row3 col8" >7</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col9" class="data row3 col9" >0.25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col11" class="data row3 col11" >12.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col12" class="data row3 col12" >1.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col13" class="data row3 col13" >0.125</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row3_col14" class="data row3 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC6</th>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col0" class="data row4 col0" >2.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col1" class="data row4 col1" >6.25</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col2" class="data row4 col2" >3</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col3" class="data row4 col3" >9</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col4" class="data row4 col4" >13</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col5" class="data row4 col5" >169</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col6" class="data row4 col6" >1</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col7" class="data row4 col7" >2.5</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col8" class="data row4 col8" >3</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col9" class="data row4 col9" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col10" class="data row4 col10" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col11" class="data row4 col11" >13</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col12" class="data row4 col12" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col13" class="data row4 col13" >0</td>
                        <td id="T_c48c449e_66ce_11e9_9032_c3aaa83f8744row4_col14" class="data row4 col14" >0</td>
            </tr>
    </tbody></table>





<style  type="text/css" >
    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col0 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col1 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col2 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col3 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col4 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col5 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col6 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col7 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col8 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col9 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col10 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col11 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col12 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col13 {
            background:  lightgreen;
        }    #T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col14 {
            background:  lightgreen;
        }</style><table id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC2</th>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col0" class="data row0 col0" >8</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col1" class="data row0 col1" >22</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col2" class="data row0 col2" >5.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col3" class="data row0 col3" >11.25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col4" class="data row0 col4" >18</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col5" class="data row0 col5" >110</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col6" class="data row0 col6" >3</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col7" class="data row0 col7" >2.66667</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col8" class="data row0 col8" >1.83333</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col9" class="data row0 col9" >0.471405</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col10" class="data row0 col10" >0.62361</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col11" class="data row0 col11" >6</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col12" class="data row0 col12" >0.816497</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col13" class="data row0 col13" >0.547507</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row0_col14" class="data row0 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC3</th>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col0" class="data row1 col0" >8.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col1" class="data row1 col1" >18.25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col2" class="data row1 col2" >28</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col3" class="data row1 col3" >198</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col4" class="data row1 col4" >39</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col5" class="data row1 col5" >389</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col6" class="data row1 col6" >4</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col7" class="data row1 col7" >2.125</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col8" class="data row1 col8" >7</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col9" class="data row1 col9" >0.216506</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col10" class="data row1 col10" >0.707107</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col11" class="data row1 col11" >9.75</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col12" class="data row1 col12" >1.47902</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col13" class="data row1 col13" >0.461807</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row1_col14" class="data row1 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC4</th>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col0" class="data row2 col0" >7.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col1" class="data row2 col1" >28.25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col2" class="data row2 col2" >14</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col3" class="data row2 col3" >98</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col4" class="data row2 col4" >25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col5" class="data row2 col5" >317</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col6" class="data row2 col6" >2</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col7" class="data row2 col7" >3.75</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col9" class="data row2 col9" >0.25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col10" class="data row2 col10" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col11" class="data row2 col11" >12.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col12" class="data row2 col12" >1.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col13" class="data row2 col13" >0.125</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row2_col14" class="data row2 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC6</th>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col0" class="data row3 col0" >2.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col1" class="data row3 col1" >6.25</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col2" class="data row3 col2" >3</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col3" class="data row3 col3" >9</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col4" class="data row3 col4" >13</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col5" class="data row3 col5" >169</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col7" class="data row3 col7" >2.5</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col8" class="data row3 col8" >3</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col11" class="data row3 col11" >13</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC7</th>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col0" class="data row4 col0" >7</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col1" class="data row4 col1" >49</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col2" class="data row4 col2" >8</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col3" class="data row4 col3" >64</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col4" class="data row4 col4" >15</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col5" class="data row4 col5" >225</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col6" class="data row4 col6" >1</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col7" class="data row4 col7" >7</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col8" class="data row4 col8" >8</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col9" class="data row4 col9" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col10" class="data row4 col10" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col11" class="data row4 col11" >15</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col12" class="data row4 col12" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col13" class="data row4 col13" >0</td>
                        <td id="T_c49d0b1c_66ce_11e9_913c_c3aaa83f8744row4_col14" class="data row4 col14" >0</td>
            </tr>
    </tbody></table>



# TIMEPOINT 16


```python
timepoint = 16
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(6-0.8, 7+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC7']['Center(X)'], cft_combined.loc['MC7']['Center(Y)']),
                        3.400368, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_29_0.png)


### New point at T16: (6, 7)
### The point falls inside the Max Boundary of the closest Micro-Cluster MC7, so it is absorbed.
### NOTE: Max Boundary of a Micro-Cluster with only 1 data point is the distance to the closest Micro-Cluster


```python
display(cft_combined)

# Add the new point
new_point = list(online.loc[[timepoint]].iloc[0])
add_to_mc('MC7', new_point[0], new_point[1], timepoint)

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC7'
                              else '' for i in x], axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3.0</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.00</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>8.5</td>
      <td>18.25</td>
      <td>28.0</td>
      <td>198.00</td>
      <td>39.0</td>
      <td>389.0</td>
      <td>4.0</td>
      <td>2.125000</td>
      <td>7.000000</td>
      <td>0.216506</td>
      <td>0.707107</td>
      <td>9.75</td>
      <td>1.479020</td>
      <td>0.461807</td>
      <td>0.923613</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>7.5</td>
      <td>28.25</td>
      <td>14.0</td>
      <td>98.00</td>
      <td>25.0</td>
      <td>317.0</td>
      <td>2.0</td>
      <td>3.750000</td>
      <td>7.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>12.50</td>
      <td>1.500000</td>
      <td>0.125000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>MC6</th>
      <td>2.5</td>
      <td>6.25</td>
      <td>3.0</td>
      <td>9.00</td>
      <td>13.0</td>
      <td>169.0</td>
      <td>1.0</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC7</th>
      <td>7.0</td>
      <td>49.00</td>
      <td>8.0</td>
      <td>64.00</td>
      <td>15.0</td>
      <td>225.0</td>
      <td>1.0</td>
      <td>7.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>





<style  type="text/css" >
    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col0 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col1 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col2 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col3 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col4 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col5 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col6 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col7 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col8 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col9 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col10 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col11 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col12 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col13 {
            background:  lightgreen;
        }    #T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col14 {
            background:  lightgreen;
        }</style><table id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC2</th>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col0" class="data row0 col0" >8</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col1" class="data row0 col1" >22</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col2" class="data row0 col2" >5.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col3" class="data row0 col3" >11.25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col4" class="data row0 col4" >18</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col5" class="data row0 col5" >110</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col6" class="data row0 col6" >3</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col7" class="data row0 col7" >2.66667</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col8" class="data row0 col8" >1.83333</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col9" class="data row0 col9" >0.471405</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col10" class="data row0 col10" >0.62361</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col11" class="data row0 col11" >6</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col12" class="data row0 col12" >0.816497</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col13" class="data row0 col13" >0.547507</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row0_col14" class="data row0 col14" >1.09501</td>
            </tr>
            <tr>
                        <th id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC3</th>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col0" class="data row1 col0" >8.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col1" class="data row1 col1" >18.25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col2" class="data row1 col2" >28</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col3" class="data row1 col3" >198</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col4" class="data row1 col4" >39</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col5" class="data row1 col5" >389</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col6" class="data row1 col6" >4</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col7" class="data row1 col7" >2.125</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col8" class="data row1 col8" >7</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col9" class="data row1 col9" >0.216506</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col10" class="data row1 col10" >0.707107</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col11" class="data row1 col11" >9.75</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col12" class="data row1 col12" >1.47902</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col13" class="data row1 col13" >0.461807</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row1_col14" class="data row1 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC4</th>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col0" class="data row2 col0" >7.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col1" class="data row2 col1" >28.25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col2" class="data row2 col2" >14</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col3" class="data row2 col3" >98</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col4" class="data row2 col4" >25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col5" class="data row2 col5" >317</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col6" class="data row2 col6" >2</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col7" class="data row2 col7" >3.75</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col9" class="data row2 col9" >0.25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col10" class="data row2 col10" >0</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col11" class="data row2 col11" >12.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col12" class="data row2 col12" >1.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col13" class="data row2 col13" >0.125</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row2_col14" class="data row2 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC6</th>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col0" class="data row3 col0" >2.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col1" class="data row3 col1" >6.25</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col2" class="data row3 col2" >3</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col3" class="data row3 col3" >9</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col4" class="data row3 col4" >13</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col5" class="data row3 col5" >169</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col7" class="data row3 col7" >2.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col8" class="data row3 col8" >3</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col11" class="data row3 col11" >13</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC7</th>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col0" class="data row4 col0" >13</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col1" class="data row4 col1" >85</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col2" class="data row4 col2" >15</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col3" class="data row4 col3" >113</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col4" class="data row4 col4" >31</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col5" class="data row4 col5" >481</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col6" class="data row4 col6" >2</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col7" class="data row4 col7" >6.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col8" class="data row4 col8" >7.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col9" class="data row4 col9" >0.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col10" class="data row4 col10" >0.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col11" class="data row4 col11" >15.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col12" class="data row4 col12" >0.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col13" class="data row4 col13" >0.5</td>
                        <td id="T_c87fe57a_66ce_11e9_aa31_c3aaa83f8744row4_col14" class="data row4 col14" >1</td>
            </tr>
    </tbody></table>



# TIMEPOINT 17


```python
timepoint = 17
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(2.5-0.8, 2+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC2']['Center(X)'], cft_combined.loc['MC2']['Center(Y)']),
                        1.09501, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_33_0.png)


### New point at T17: (2.5, 2)
### The point falls inside the Max Boundary of the closest Micro-Cluster MC2, so it is absorbed.


```python
display(cft_combined)

# Add the new point
new_point = list(online.loc[[timepoint]].iloc[0])
add_to_mc('MC2', new_point[0], new_point[1], timepoint)

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC2'
                              else '' for i in x], axis=1)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CF1(X)</th>
      <th>CF2(X)</th>
      <th>CF1(Y)</th>
      <th>CF2(Y)</th>
      <th>CF1(T)</th>
      <th>CF2(T)</th>
      <th>N</th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Radius(X)</th>
      <th>Radius(Y)</th>
      <th>Mean(T)</th>
      <th>Sigma(T)</th>
      <th>Radius</th>
      <th>Max Radius</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC2</th>
      <td>8.0</td>
      <td>22.00</td>
      <td>5.5</td>
      <td>11.25</td>
      <td>18.0</td>
      <td>110.0</td>
      <td>3.0</td>
      <td>2.666667</td>
      <td>1.833333</td>
      <td>0.471405</td>
      <td>0.623610</td>
      <td>6.00</td>
      <td>0.816497</td>
      <td>0.547507</td>
      <td>1.095014</td>
    </tr>
    <tr>
      <th>MC3</th>
      <td>8.5</td>
      <td>18.25</td>
      <td>28.0</td>
      <td>198.00</td>
      <td>39.0</td>
      <td>389.0</td>
      <td>4.0</td>
      <td>2.125000</td>
      <td>7.000000</td>
      <td>0.216506</td>
      <td>0.707107</td>
      <td>9.75</td>
      <td>1.479020</td>
      <td>0.461807</td>
      <td>0.923613</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>7.5</td>
      <td>28.25</td>
      <td>14.0</td>
      <td>98.00</td>
      <td>25.0</td>
      <td>317.0</td>
      <td>2.0</td>
      <td>3.750000</td>
      <td>7.000000</td>
      <td>0.250000</td>
      <td>0.000000</td>
      <td>12.50</td>
      <td>1.500000</td>
      <td>0.125000</td>
      <td>0.250000</td>
    </tr>
    <tr>
      <th>MC6</th>
      <td>2.5</td>
      <td>6.25</td>
      <td>3.0</td>
      <td>9.00</td>
      <td>13.0</td>
      <td>169.0</td>
      <td>1.0</td>
      <td>2.500000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MC7</th>
      <td>13.0</td>
      <td>85.00</td>
      <td>15.0</td>
      <td>113.00</td>
      <td>31.0</td>
      <td>481.0</td>
      <td>2.0</td>
      <td>6.500000</td>
      <td>7.500000</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>15.50</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>





<style  type="text/css" >
    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col0 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col1 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col2 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col3 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col4 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col5 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col6 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col7 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col8 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col9 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col10 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col11 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col12 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col13 {
            background:  lightgreen;
        }    #T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col14 {
            background:  lightgreen;
        }</style><table id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC2</th>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col0" class="data row0 col0" >10.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col1" class="data row0 col1" >28.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col2" class="data row0 col2" >7.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col3" class="data row0 col3" >15.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col4" class="data row0 col4" >35</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col5" class="data row0 col5" >399</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col6" class="data row0 col6" >4</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col7" class="data row0 col7" >2.625</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col8" class="data row0 col8" >1.875</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col9" class="data row0 col9" >0.414578</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col10" class="data row0 col10" >0.544862</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col11" class="data row0 col11" >8.75</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col12" class="data row0 col12" >4.81534</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col13" class="data row0 col13" >0.47972</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row0_col14" class="data row0 col14" >0.95944</td>
            </tr>
            <tr>
                        <th id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC3</th>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col0" class="data row1 col0" >8.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col1" class="data row1 col1" >18.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col2" class="data row1 col2" >28</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col3" class="data row1 col3" >198</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col4" class="data row1 col4" >39</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col5" class="data row1 col5" >389</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col6" class="data row1 col6" >4</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col7" class="data row1 col7" >2.125</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col8" class="data row1 col8" >7</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col9" class="data row1 col9" >0.216506</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col10" class="data row1 col10" >0.707107</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col11" class="data row1 col11" >9.75</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col12" class="data row1 col12" >1.47902</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col13" class="data row1 col13" >0.461807</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row1_col14" class="data row1 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC4</th>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col0" class="data row2 col0" >7.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col1" class="data row2 col1" >28.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col2" class="data row2 col2" >14</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col3" class="data row2 col3" >98</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col4" class="data row2 col4" >25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col5" class="data row2 col5" >317</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col6" class="data row2 col6" >2</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col7" class="data row2 col7" >3.75</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col9" class="data row2 col9" >0.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col10" class="data row2 col10" >0</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col11" class="data row2 col11" >12.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col12" class="data row2 col12" >1.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col13" class="data row2 col13" >0.125</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row2_col14" class="data row2 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC6</th>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col0" class="data row3 col0" >2.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col1" class="data row3 col1" >6.25</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col2" class="data row3 col2" >3</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col3" class="data row3 col3" >9</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col4" class="data row3 col4" >13</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col5" class="data row3 col5" >169</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col7" class="data row3 col7" >2.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col8" class="data row3 col8" >3</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col11" class="data row3 col11" >13</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC7</th>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col0" class="data row4 col0" >13</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col1" class="data row4 col1" >85</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col2" class="data row4 col2" >15</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col3" class="data row4 col3" >113</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col4" class="data row4 col4" >31</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col5" class="data row4 col5" >481</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col6" class="data row4 col6" >2</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col7" class="data row4 col7" >6.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col8" class="data row4 col8" >7.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col9" class="data row4 col9" >0.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col10" class="data row4 col10" >0.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col11" class="data row4 col11" >15.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col12" class="data row4 col12" >0.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col13" class="data row4 col13" >0.5</td>
                        <td id="T_c9eb012c_66ce_11e9_925f_c3aaa83f8744row4_col14" class="data row4 col14" >1</td>
            </tr>
    </tbody></table>



# TIMEPOINT 18


```python
timepoint = 18
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)


# New point
p1 = sns.scatterplot(x="X", y="Y", data=online.loc[[timepoint]], s=50)
p1.text(5-0.8, 5+0.15, "New Point", horizontalalignment='left', size='medium', color='black', weight='semibold')


# Plot the micro cluster radius
ax.add_patch(plt.Circle((cft_combined.loc['MC4']['Center(X)'], cft_combined.loc['MC4']['Center(Y)']),
                        0.25, color='r', alpha=0.5))

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_37_0.png)


### New point at T18: (5, 5)
### The point falls outside the Max Boundary of the closest Micro-Cluster MC4, so a new Micro-Cluster has to be created.
### In order to accommodate the new Micro-Cluster, we first try to delete a Micro-Cluster. Currently, all Micro-Clusters fulfill the relevency threshold, so two of the closest Micro-Clusters should be merged. 
### MC2 and MC6 are currently the closest


```python
display(cft_combined.copy().style.apply(lambda x: ['background: lightcoral' if x.name == 'MC2' or x.name == 'MC6'
                                                   else '' for i in x], axis=1))

cft_combined = merge_mc(mc1='MC2', mc2='MC6', new_mc='MC8')
recalculate_summaries()

# Add the new point
new_point = list(online.loc[[timepoint]].iloc[0]) 
cft_combined = create_new_mc(new_point[0], new_point[1], timepoint, 'MC9')

# Recalculate cluster summaries
recalculate_summaries()
cft_combined.copy().style.apply(lambda x: ['background: lightgreen' if x.name == 'MC9'
                              else '' for i in x], axis=1)
```


<style  type="text/css" >
    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col0 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col1 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col2 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col3 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col4 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col5 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col6 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col7 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col8 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col9 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col10 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col11 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col12 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col13 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col14 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col0 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col1 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col2 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col3 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col4 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col5 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col6 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col7 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col8 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col9 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col10 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col11 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col12 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col13 {
            background:  lightcoral;
        }    #T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col14 {
            background:  lightcoral;
        }</style><table id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC2</th>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col0" class="data row0 col0" >10.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col1" class="data row0 col1" >28.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col2" class="data row0 col2" >7.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col3" class="data row0 col3" >15.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col4" class="data row0 col4" >35</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col5" class="data row0 col5" >399</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col6" class="data row0 col6" >4</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col7" class="data row0 col7" >2.625</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col8" class="data row0 col8" >1.875</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col9" class="data row0 col9" >0.414578</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col10" class="data row0 col10" >0.544862</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col11" class="data row0 col11" >8.75</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col12" class="data row0 col12" >4.81534</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col13" class="data row0 col13" >0.47972</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row0_col14" class="data row0 col14" >0.95944</td>
            </tr>
            <tr>
                        <th id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC3</th>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col0" class="data row1 col0" >8.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col1" class="data row1 col1" >18.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col2" class="data row1 col2" >28</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col3" class="data row1 col3" >198</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col4" class="data row1 col4" >39</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col5" class="data row1 col5" >389</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col6" class="data row1 col6" >4</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col7" class="data row1 col7" >2.125</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col8" class="data row1 col8" >7</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col9" class="data row1 col9" >0.216506</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col10" class="data row1 col10" >0.707107</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col11" class="data row1 col11" >9.75</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col12" class="data row1 col12" >1.47902</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col13" class="data row1 col13" >0.461807</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row1_col14" class="data row1 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC4</th>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col0" class="data row2 col0" >7.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col1" class="data row2 col1" >28.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col2" class="data row2 col2" >14</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col3" class="data row2 col3" >98</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col4" class="data row2 col4" >25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col5" class="data row2 col5" >317</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col6" class="data row2 col6" >2</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col7" class="data row2 col7" >3.75</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col8" class="data row2 col8" >7</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col9" class="data row2 col9" >0.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col10" class="data row2 col10" >0</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col11" class="data row2 col11" >12.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col12" class="data row2 col12" >1.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col13" class="data row2 col13" >0.125</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row2_col14" class="data row2 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC6</th>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col0" class="data row3 col0" >2.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col1" class="data row3 col1" >6.25</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col2" class="data row3 col2" >3</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col3" class="data row3 col3" >9</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col4" class="data row3 col4" >13</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col5" class="data row3 col5" >169</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col6" class="data row3 col6" >1</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col7" class="data row3 col7" >2.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col8" class="data row3 col8" >3</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col9" class="data row3 col9" >0</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col10" class="data row3 col10" >0</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col11" class="data row3 col11" >13</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col12" class="data row3 col12" >0</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col13" class="data row3 col13" >0</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row3_col14" class="data row3 col14" >0</td>
            </tr>
            <tr>
                        <th id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC7</th>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col0" class="data row4 col0" >13</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col1" class="data row4 col1" >85</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col2" class="data row4 col2" >15</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col3" class="data row4 col3" >113</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col4" class="data row4 col4" >31</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col5" class="data row4 col5" >481</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col6" class="data row4 col6" >2</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col7" class="data row4 col7" >6.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col8" class="data row4 col8" >7.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col9" class="data row4 col9" >0.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col10" class="data row4 col10" >0.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col11" class="data row4 col11" >15.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col12" class="data row4 col12" >0.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col13" class="data row4 col13" >0.5</td>
                        <td id="T_cb9ceb1c_66ce_11e9_8416_c3aaa83f8744row4_col14" class="data row4 col14" >1</td>
            </tr>
    </tbody></table>





<style  type="text/css" >
    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col0 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col1 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col2 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col3 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col4 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col5 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col6 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col7 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col8 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col9 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col10 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col11 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col12 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col13 {
            background:  lightgreen;
        }    #T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col14 {
            background:  lightgreen;
        }</style><table id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >CF1(X)</th>        <th class="col_heading level0 col1" >CF2(X)</th>        <th class="col_heading level0 col2" >CF1(Y)</th>        <th class="col_heading level0 col3" >CF2(Y)</th>        <th class="col_heading level0 col4" >CF1(T)</th>        <th class="col_heading level0 col5" >CF2(T)</th>        <th class="col_heading level0 col6" >N</th>        <th class="col_heading level0 col7" >Center(X)</th>        <th class="col_heading level0 col8" >Center(Y)</th>        <th class="col_heading level0 col9" >Radius(X)</th>        <th class="col_heading level0 col10" >Radius(Y)</th>        <th class="col_heading level0 col11" >Mean(T)</th>        <th class="col_heading level0 col12" >Sigma(T)</th>        <th class="col_heading level0 col13" >Radius</th>        <th class="col_heading level0 col14" >Max Radius</th>    </tr>    <tr>        <th class="index_name level0" >MicroCluster</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744level0_row0" class="row_heading level0 row0" >MC3</th>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col0" class="data row0 col0" >8.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col1" class="data row0 col1" >18.25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col2" class="data row0 col2" >28</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col3" class="data row0 col3" >198</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col4" class="data row0 col4" >39</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col5" class="data row0 col5" >389</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col6" class="data row0 col6" >4</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col7" class="data row0 col7" >2.125</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col8" class="data row0 col8" >7</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col9" class="data row0 col9" >0.216506</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col10" class="data row0 col10" >0.707107</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col11" class="data row0 col11" >9.75</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col12" class="data row0 col12" >1.47902</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col13" class="data row0 col13" >0.461807</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row0_col14" class="data row0 col14" >0.923613</td>
            </tr>
            <tr>
                        <th id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744level0_row1" class="row_heading level0 row1" >MC4</th>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col0" class="data row1 col0" >7.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col1" class="data row1 col1" >28.25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col2" class="data row1 col2" >14</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col3" class="data row1 col3" >98</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col4" class="data row1 col4" >25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col5" class="data row1 col5" >317</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col6" class="data row1 col6" >2</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col7" class="data row1 col7" >3.75</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col8" class="data row1 col8" >7</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col9" class="data row1 col9" >0.25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col10" class="data row1 col10" >0</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col11" class="data row1 col11" >12.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col12" class="data row1 col12" >1.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col13" class="data row1 col13" >0.125</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row1_col14" class="data row1 col14" >0.25</td>
            </tr>
            <tr>
                        <th id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744level0_row2" class="row_heading level0 row2" >MC7</th>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col0" class="data row2 col0" >13</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col1" class="data row2 col1" >85</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col2" class="data row2 col2" >15</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col3" class="data row2 col3" >113</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col4" class="data row2 col4" >31</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col5" class="data row2 col5" >481</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col6" class="data row2 col6" >2</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col7" class="data row2 col7" >6.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col8" class="data row2 col8" >7.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col9" class="data row2 col9" >0.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col10" class="data row2 col10" >0.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col11" class="data row2 col11" >15.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col12" class="data row2 col12" >0.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col13" class="data row2 col13" >0.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row2_col14" class="data row2 col14" >1</td>
            </tr>
            <tr>
                        <th id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744level0_row3" class="row_heading level0 row3" >MC8</th>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col0" class="data row3 col0" >13</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col1" class="data row3 col1" >34.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col2" class="data row3 col2" >10.5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col3" class="data row3 col3" >24.25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col4" class="data row3 col4" >48</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col5" class="data row3 col5" >568</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col6" class="data row3 col6" >5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col7" class="data row3 col7" >2.6</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col8" class="data row3 col8" >2.1</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col9" class="data row3 col9" >0.374166</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col10" class="data row3 col10" >0.663325</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col11" class="data row3 col11" >9.6</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col12" class="data row3 col12" >4.63033</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col13" class="data row3 col13" >0.518745</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row3_col14" class="data row3 col14" >1.03749</td>
            </tr>
            <tr>
                        <th id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744level0_row4" class="row_heading level0 row4" >MC9</th>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col0" class="data row4 col0" >5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col1" class="data row4 col1" >25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col2" class="data row4 col2" >5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col3" class="data row4 col3" >25</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col4" class="data row4 col4" >18</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col5" class="data row4 col5" >324</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col6" class="data row4 col6" >1</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col7" class="data row4 col7" >5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col8" class="data row4 col8" >5</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col9" class="data row4 col9" >0</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col10" class="data row4 col10" >0</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col11" class="data row4 col11" >18</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col12" class="data row4 col12" >0</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col13" class="data row4 col13" >0</td>
                        <td id="T_cbaf11a8_66ce_11e9_bf88_c3aaa83f8744row4_col14" class="data row4 col14" >0</td>
            </tr>
    </tbody></table>



### We reached the end of the stream. The final Micro-Clusters are plotted below.


```python
# Plot the current micro clusters
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="MicroCluster", hue="MicroCluster", data=cft_combined.reset_index(), s=150)

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


![png](output_41_0.png)


### Now, we run an offline K-Means algorithm with k centroids, in order to get k final clusters.
### The centers are initialized proportional to the number of points in a given microcluster.
### We are manually initializing the centroids to the values below:


```python
d = {'Centers': ['C1', 'C2', 'C3',], 'X': [3, 3.5, 6], 'Y': [3, 6, 6.5]}
weighted_centroids = pd.DataFrame(d).set_index('Centers')
weighted_centroids
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
    </tr>
    <tr>
      <th>Centers</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C1</th>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>C2</th>
      <td>3.5</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>C3</th>
      <td>6.0</td>
      <td>6.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
microclusters = cft_combined[['Center(X)', 'Center(Y)']]
kmeans = KMeans(n_clusters=3, random_state=0, init=weighted_centroids)
kmeans.fit(microclusters)

labels = kmeans.labels_
# Change the cluster labels from just numbers like 0, 1, 2 to MC1, MC2, etc
labels = ['MC'+str((x+1)) for x in labels]
microclusters['Centroid'] = labels
display(microclusters)

# Plot the K-means output with assignments
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
sns.scatterplot(x="Center(X)", y="Center(Y)", style="Centroid", hue="Centroid", data=microclusters, s=150)

#Use adjustable='box-forced' to make the plot area square-shaped as well.
ax.set_aspect('equal', adjustable='datalim')
ax.plot()   #Causes an autoscale update.
plt.show()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Center(X)</th>
      <th>Center(Y)</th>
      <th>Centroid</th>
    </tr>
    <tr>
      <th>MicroCluster</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MC3</th>
      <td>2.125</td>
      <td>7.0</td>
      <td>MC2</td>
    </tr>
    <tr>
      <th>MC4</th>
      <td>3.750</td>
      <td>7.0</td>
      <td>MC2</td>
    </tr>
    <tr>
      <th>MC7</th>
      <td>6.500</td>
      <td>7.5</td>
      <td>MC3</td>
    </tr>
    <tr>
      <th>MC8</th>
      <td>2.600</td>
      <td>2.1</td>
      <td>MC1</td>
    </tr>
    <tr>
      <th>MC9</th>
      <td>5.000</td>
      <td>5.0</td>
      <td>MC2</td>
    </tr>
  </tbody>
</table>
</div>



![png](output_44_1.png)

