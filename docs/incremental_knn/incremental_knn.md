

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.metrics import accuracy_score, cohen_kappa_score
```


```python
def jaccard_similarity(x, y):
    """
    Returns jaccard score between x and y
    """
    return np.logical_and(x, y).sum() / np.logical_or(x, y).sum()
```


```python
d = {'TS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Text': ['Regularly paying too much for free trials?', 
             'Exercise as a chance for your free vehicle.',
             'I have just as much fun as I need.',
             'Do you like donuts?',
             'Fresh donuts available for cheap',
             'They had fresh donuts available, so today was fun',
             'Register your free trial today',
             'What time is good for you?',
             'I didn\'t pay for the donuts',
             'Cheap viagra available',
             'Did you have a good time today?',
             'It was available so I registered'], 
     'Transformed': ['regular pay free trial', 
                     'exercise chance free vehicle',
                     'fun need',
                     'like donut',
                     'fresh donut available cheap',
                     'fresh donut available today fun',
                     'register free trial today',
                     'time good',
                     'pay donut',
                     'cheap viagra available',
                     'good time today',
                     'available register'],
     'Class': ['Spam', 'Spam', 'Not spam', 'Not spam', 'Spam', 'Not spam', 'Spam', 'Not spam', 'Not spam', 'Spam', 'Not spam', 'Not spam']}
data = pd.DataFrame(d)
```

# Data stream of documents


```python
display(data[['TS', 'Text', 'Class']])
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
      <th>TS</th>
      <th>Text</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Regularly paying too much for free trials?</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Exercise as a chance for your free vehicle.</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>I have just as much fun as I need.</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Do you like donuts?</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Fresh donuts available for cheap</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>They had fresh donuts available, so today was fun</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Register your free trial today</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>What time is good for you?</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>I didn't pay for the donuts</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Cheap viagra available</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>Did you have a good time today?</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>It was available so I registered</td>
      <td>Not spam</td>
    </tr>
  </tbody>
</table>
</div>


## Convert  the  texts  into  binary  vectors  where  the  presence  of  a  term  is 1  and  the  absence  is  0.   Use  the  following  structure  for  the  document vectors,  which  excludes  stop  words: 
## *[regular,  pay,  free,  trial,  exercise,chance,  vehicle,  fun,  need,  like,  donut,  fresh,  available,  cheap,  register, today,  time,  good,  viagra,  run]*  
### Note:  Assume  there  is  a  pre-processing function that stems the terms, so paying becomes pay, trials become trial,etc.


```python
vocab = ['regular', 'pay', 'free', 'trial', 'exercise', 'chance', 'vehicle', 'fun', 'need', 'like', 'donut', 'fresh', 'available', 'cheap', 'register', 'today', 'time', 'good', 'viagra', 'run']
vec = CountVectorizer(binary=True, stop_words='english', lowercase=True, vocabulary=vocab)
X = vec.fit_transform(data.Transformed)
text_vectors = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
text_vectors['TS'] = data.TS
text_vectors.set_index('TS', inplace=True)
display(text_vectors)
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
      <th>regular</th>
      <th>pay</th>
      <th>free</th>
      <th>trial</th>
      <th>exercise</th>
      <th>chance</th>
      <th>vehicle</th>
      <th>fun</th>
      <th>need</th>
      <th>like</th>
      <th>donut</th>
      <th>fresh</th>
      <th>available</th>
      <th>cheap</th>
      <th>register</th>
      <th>today</th>
      <th>time</th>
      <th>good</th>
      <th>viagra</th>
      <th>run</th>
    </tr>
    <tr>
      <th>TS</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## To measure document similarity, calculate the Jaccard coefficient between two document vectors.
## Example: 
### doc1=[1, 1, 0, 0] 
### doc2=[1, 0, 1, 0]
### Jaccard(doc1, doc2) = intersection / union
### intersection: number of times 1 appears in both docs at the same position (1)
### union: number of times 1 appears in one vector and either 0 or 1 appears in the other (3)
### 1 / (1 + 1 + 1) = 0.66

## TS 6
### Window: [1, 2, 3, 4, 5]


```python
print('Jaccard(6, 1):', jaccard_similarity(text_vectors.loc[6], text_vectors.loc[1]))
print('Jaccard(6, 2):', jaccard_similarity(text_vectors.loc[6], text_vectors.loc[2]))
print('Jaccard(6, 3):', jaccard_similarity(text_vectors.loc[6], text_vectors.loc[3]))
print('Jaccard(6, 4):', jaccard_similarity(text_vectors.loc[6], text_vectors.loc[4]))
print('Jaccard(6, 5):', jaccard_similarity(text_vectors.loc[6], text_vectors.loc[5]))
```

    Jaccard(6, 1): 0.0
    Jaccard(6, 2): 0.0
    Jaccard(6, 3): 0.16666666666666666
    Jaccard(6, 4): 0.16666666666666666
    Jaccard(6, 5): 0.5
    

### Nearest Neighbors: [3: Not spam, 4: Not spam, 5: Spam]
### Classification: Not spam

## TS 7
### Window: [2, 3, 4, 5, 6]


```python
print('Jaccard(7, 2):', jaccard_similarity(text_vectors.loc[7], text_vectors.loc[2]))
print('Jaccard(7, 3):', jaccard_similarity(text_vectors.loc[7], text_vectors.loc[3]))
print('Jaccard(7, 4):', jaccard_similarity(text_vectors.loc[7], text_vectors.loc[4]))
print('Jaccard(7, 5):', jaccard_similarity(text_vectors.loc[7], text_vectors.loc[5]))
print('Jaccard(7, 6):', jaccard_similarity(text_vectors.loc[7], text_vectors.loc[6]))
```

    Jaccard(7, 2): 0.14285714285714285
    Jaccard(7, 3): 0.0
    Jaccard(7, 4): 0.0
    Jaccard(7, 5): 0.0
    Jaccard(7, 6): 0.125
    

### Nearest Neighbors: [2: Spam, 6: Not spam, 3, 4, 5] (Since there are many instances tied for 3rd nearest neighbor, keep lowering K till the tie is broken)
### Nearest Neighbor: [2: Spam]
### Classification: Spam

## TS 8
### Window: [3, 4, 5, 6, 7]


```python
print('Jaccard(8, 3):', jaccard_similarity(text_vectors.loc[8], text_vectors.loc[3]))
print('Jaccard(8, 4):', jaccard_similarity(text_vectors.loc[8], text_vectors.loc[4]))
print('Jaccard(8, 5):', jaccard_similarity(text_vectors.loc[8], text_vectors.loc[5]))
print('Jaccard(8, 6):', jaccard_similarity(text_vectors.loc[8], text_vectors.loc[6]))
print('Jaccard(8, 7):', jaccard_similarity(text_vectors.loc[8], text_vectors.loc[7]))
```

    Jaccard(8, 3): 0.0
    Jaccard(8, 4): 0.0
    Jaccard(8, 5): 0.0
    Jaccard(8, 6): 0.0
    Jaccard(8, 7): 0.0
    

### Since all instances are tied, use majority classification
### Classification: Not spam

## TS 9
### Window: [4, 5, 6, 7, 8]


```python
print('Jaccard(9, 4):', jaccard_similarity(text_vectors.loc[9], text_vectors.loc[4]))
print('Jaccard(9, 5):', jaccard_similarity(text_vectors.loc[9], text_vectors.loc[5]))
print('Jaccard(9, 6):', jaccard_similarity(text_vectors.loc[9], text_vectors.loc[6]))
print('Jaccard(9, 7):', jaccard_similarity(text_vectors.loc[9], text_vectors.loc[7]))
print('Jaccard(9, 8):', jaccard_similarity(text_vectors.loc[9], text_vectors.loc[8]))
```

    Jaccard(9, 4): 0.3333333333333333
    Jaccard(9, 5): 0.2
    Jaccard(9, 6): 0.16666666666666666
    Jaccard(9, 7): 0.0
    Jaccard(9, 8): 0.0
    

### Nearest Neighbors: [4: Not Spam, 6: Not spam, 5, 7, 8] (Since there are many instances tied for 3rd nearest neighbor, keep lowering K till the tie is broken)
### Nearest Neighbor: [4: Not spam, 6: Not spam]
### Classification: Not spam

## TS 10
### Window: [5, 6, 7, 8, 9]


```python
print('Jaccard(10, 5):', jaccard_similarity(text_vectors.loc[10], text_vectors.loc[5]))
print('Jaccard(10, 6):', jaccard_similarity(text_vectors.loc[10], text_vectors.loc[6]))
print('Jaccard(10, 7):', jaccard_similarity(text_vectors.loc[10], text_vectors.loc[7]))
print('Jaccard(10, 8):', jaccard_similarity(text_vectors.loc[10], text_vectors.loc[8]))
print('Jaccard(10, 9):', jaccard_similarity(text_vectors.loc[10], text_vectors.loc[9]))
```

    Jaccard(10, 5): 0.4
    Jaccard(10, 6): 0.14285714285714285
    Jaccard(10, 7): 0.0
    Jaccard(10, 8): 0.0
    Jaccard(10, 9): 0.0
    

### Nearest Neighbors: [5: Spam, 10: Spam, 7, 8, 9] (Since there are many instances tied for 3rd nearest neighbor, keep lowering K till the tie is broken)
### Nearest Neighbor: [5: Spam, 10: Spam]
### Classification: Spam

## TS 11
### Window: [6, 7, 8, 9, 10]


```python
print('Jaccard(11, 6):', jaccard_similarity(text_vectors.loc[11], text_vectors.loc[6]))
print('Jaccard(11, 7):', jaccard_similarity(text_vectors.loc[11], text_vectors.loc[7]))
print('Jaccard(11, 8):', jaccard_similarity(text_vectors.loc[11], text_vectors.loc[8]))
print('Jaccard(11, 9):', jaccard_similarity(text_vectors.loc[11], text_vectors.loc[9]))
print('Jaccard(11, 10):', jaccard_similarity(text_vectors.loc[11], text_vectors.loc[10]))
```

    Jaccard(11, 6): 0.14285714285714285
    Jaccard(11, 7): 0.16666666666666666
    Jaccard(11, 8): 0.6666666666666666
    Jaccard(11, 9): 0.0
    Jaccard(11, 10): 0.0
    

### Nearest Neighbors: [8: Not spam, 7: Spam, 6: Not spam]
### Classification: Not spam

## TS 12
### Window: [7, 8, 9, 10, 11]


```python
print('Jaccard(12, 7):', jaccard_similarity(text_vectors.loc[12], text_vectors.loc[7]))
print('Jaccard(12, 8):', jaccard_similarity(text_vectors.loc[12], text_vectors.loc[8]))
print('Jaccard(12, 9):', jaccard_similarity(text_vectors.loc[12], text_vectors.loc[9]))
print('Jaccard(12, 10):', jaccard_similarity(text_vectors.loc[12], text_vectors.loc[10]))
print('Jaccard(12, 11):', jaccard_similarity(text_vectors.loc[12], text_vectors.loc[11]))
```

    Jaccard(12, 7): 0.2
    Jaccard(12, 8): 0.0
    Jaccard(12, 9): 0.0
    Jaccard(12, 10): 0.25
    Jaccard(12, 11): 0.0
    

### Nearest Neighbors: [10: Spam, 7: Spam, 8, 9, 10] (Since there are many instances tied for 3rd nearest neighbor, keep lowering K till the tie is broken)
### Nearest Neighbor: [10: Spam, 7: Spam]
### Classification: Spam

# Summary of predictions


```python
r = {'ts': [6, 7, 8, 9, 10, 11, 12],
     'pred': ['Not spam', 'Spam', 'Not spam', 'Not spam', 'Spam', 'Not spam', 'Spam'],
     'actual': ['Not spam', 'Spam', 'Not spam', 'Not spam', 'Spam', 'Spam', 'Not spam']}
results = pd.DataFrame(r).set_index('ts')
results
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
      <th>pred</th>
      <th>actual</th>
    </tr>
    <tr>
      <th>ts</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>Not spam</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Spam</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Not spam</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Not spam</td>
      <td>Not spam</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Spam</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Not spam</td>
      <td>Spam</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Spam</td>
      <td>Not spam</td>
    </tr>
  </tbody>
</table>
</div>



###  When the dataset is imbalanced, computing Kappa against the ground truth gives a more reliable performance estimate than accuracy. Higher values are better


```python
print('Accuracy:', np.round(accuracy_score(results.actual, results.pred), 2))
print('Kappa   :', np.round(cohen_kappa_score(results.actual, results.pred), 2))
```

    Accuracy: 0.71
    Kappa   : 0.42
    


```python

```


```python

```
