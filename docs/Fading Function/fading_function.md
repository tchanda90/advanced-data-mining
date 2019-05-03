
# Fading Function

**For a damped window model, consider the fading function f(t) = 2^−λt, where t is
the time-point and λ is a user-defined parameter. What is the weight of an instance x
observed at time-point T(T > t)? Calculate the weight of the instance x at t0, t1, t2, t3, t4
since time t0. Plot a graph of hte weight v/s the time-point.**


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```


```python
lam = 1
def f(t):
    return 2**(-lam*t)
```


```python
timepoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
weights = []
for timepoint in timepoints:
    weight = f(timepoint)
    weights.append(weight)
```


```python
df = pd.DataFrame(weights, columns=[['Weight']])
df['Timepoint'] = df.index
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>Weight</th>
      <th>Timepoint</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.500000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.250000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.125000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.062500</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.031250</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.015625</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.007812</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.003906</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.001953</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.000977</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
fig.set_size_inches(15, 8.27)
plt.title('Weight of an instance X over different timepoints according to the fading function f(t) = 2^−λt')
plt.xlabel('Time')
plt.ylabel('Weight')
plt.plot(df['Timepoint'].values, df['Weight'].values, marker='o')
```




    [<matplotlib.lines.Line2D at 0x138043f0>]




![png](output_6_1.png)


### As Time increases, the weight of the instance X gets smaller and smaller
