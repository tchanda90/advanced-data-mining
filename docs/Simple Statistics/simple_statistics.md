
### Simple Statistics

Let the following data set be given (sample size 60):

4, 3, 2, 5, 4, 6, 3, 7, 4, 1, 4, 0, 6, 4, 3, 5, 2, 3, 5, 1, 4, 4, 9, 5, 4, 3, 3, 5, 2, 4, 3, 6, 5, 2, 6, 2, 4, 5, 5, 1, 5, 4, 4, 2, 7, 1, 3, 3, 4, 7, 3, 4, 4, 6, 6, 3, 3, 2, 6, 1.

Calculate:
- The mean
- The mean recursively
- The standard deviation over the sample

The calculation of the mean and the standard deviation of a list of numbers is fairly straightforward.

![mean](mean.png)

![std](std.png)


```python
import math
import time

numbers = [4, 3, 2, 5, 4, 6, 3, 7, 4, 1, 4, 0, 6, 4, 3, 5, 2, 3, 5, 1, 4, 4, 9, 5, 4, 3, 3, 5, 2, 4, 3, 6, 5, 2, 
           6, 2, 4, 5, 5, 1, 5, 4, 4, 2, 7, 1, 3, 3, 4, 7, 3, 4, 4, 6, 6, 3, 3, 2, 6, 1]

# calculate mean
mean = sum(numbers) / len(numbers)

# calculate std dev
std_dev = math.sqrt(
    sum([(x - mean)**2 for x in numbers]) 
    / len(numbers)
)

print('Sample mean    : %0.2f' % mean)
print('Sample std dev : %0.2f' % std_dev)
```

    Sample mean    : 3.87
    Sample std dev : 1.77
    

However, in streaming environments, x is **unbounded**, which makes it necessary to calculate these simple statistics **incrementaly**.

To incrementally calculate the mean and standard deviation of a random variable x, we need to maintain three variables for x:
- **LS (Linear Sum)**
- **SS (Squared Sum)**
- **N (Count)**

This allows observations to be incrementally added.
- LS = LS + $x_{i}$
- SS = SS + $x_{i}^2$
- N = N + 1

As shown below, these three variables and their incremental additive properties are sufficient to calculate the mean and standard deviation of x in a streaming environment.


```python
class Stream:
    
    def __init__(self):
        self.ls = 0.0
        self.ss = 0.0
        self.n = 0.0
    
    def increment(self, x):
        """
        Add x to the observations by incrementing the sufficient stats
        """
        self.ls += x
        self.ss += x**2
        self.n += 1

    def decrement(self, x):
        """
        Remove x from the observations by decrementing the sufficient stats
        """
        self.ls -= x
        self.ss -= x**2
        self.n -= 1
    
    def mean(self):
        """
        Return mean of the observations by dividing LS by N
        """
        return self.ls/self.n
    
    def std_dev(self):
        """
        Return the standard deviation of the observations
        """
        return math.sqrt((self.ss/self.n) - (self.ls/self.n)**2)
    
    def print_stats(self):
        """
        Print the current values of the sufficient stats to the console
        """
        print('Linear Sum  : %0.2f' % self.ls)
        print('Squared Sum : %0.2f' % self.ss)
        print('N           : %0.2f' % self.n)
```

The mean can be calculated by:

![inc_mean](inc_mean.png)

And the standard deviation can be calculated by:

![inc_std](inc_std.png)


Below, we are incrementally adding three numbers to the sample, and calculating the mean and standard deviation of the observations in the stream


```python
stream = Stream()

stream.increment(4)
stream.increment(3)
stream.increment(2)

stream.print_stats()
print()
print('Mean: %0.2f' % stream.mean())
print('Standard Deviation: %0.2f' % stream.std_dev())
```

    Linear Sum  : 9.00
    Squared Sum : 29.00
    N           : 3.00
    
    Mean: 3.00
    Standard Deviation: 0.82
    

**Coming back to the original sample of 60 items:**

**4, 3, 2, 5, 4, 6, 3, 7, 4, 1, 4, 0, 6, 4, 3, 5, 2, 3, 5, 1, 4, 4, 9, 5, 4, 3, 3, 5, 2, 4, 3, 6, 5, 2, 6, 2, 4, 5, 5, 1, 5, 4, 4, 2, 7, 1, 3, 3, 4, 7, 3, 4, 4, 6, 6, 3, 3, 2, 6, 1.**

**Below, a stream is simulated where the items arrive one by one with some time delay. They are incrementally added to the stream by updating the sufficient statistics, then the sufficient statistics along with the running mean and standard deviation are printed.**


```python
stream = Stream()
for number in numbers:
    print('Incoming Item: %d' % number)
    stream.increment(number)
    print('[LS,    SS,    N]')
    print([stream.ls, stream.ss, stream.n])
    print()
    print('Mean: %0.2f, Std Dev: %0.2f' % (stream.mean(), stream.std_dev()))
    print('=============================')
    time.sleep(3)
```

    Incoming Item: 4
    [LS,    SS,    N]
    [4.0, 16.0, 1.0]
    
    Mean: 4.00, Std Dev: 0.00
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [7.0, 25.0, 2.0]
    
    Mean: 3.50, Std Dev: 0.50
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [9.0, 29.0, 3.0]
    
    Mean: 3.00, Std Dev: 0.82
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [14.0, 54.0, 4.0]
    
    Mean: 3.50, Std Dev: 1.12
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [18.0, 70.0, 5.0]
    
    Mean: 3.60, Std Dev: 1.02
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [24.0, 106.0, 6.0]
    
    Mean: 4.00, Std Dev: 1.29
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [27.0, 115.0, 7.0]
    
    Mean: 3.86, Std Dev: 1.25
    =============================
    Incoming Item: 7
    [LS,    SS,    N]
    [34.0, 164.0, 8.0]
    
    Mean: 4.25, Std Dev: 1.56
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [38.0, 180.0, 9.0]
    
    Mean: 4.22, Std Dev: 1.47
    =============================
    Incoming Item: 1
    [LS,    SS,    N]
    [39.0, 181.0, 10.0]
    
    Mean: 3.90, Std Dev: 1.70
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [43.0, 197.0, 11.0]
    
    Mean: 3.91, Std Dev: 1.62
    =============================
    Incoming Item: 0
    [LS,    SS,    N]
    [43.0, 197.0, 12.0]
    
    Mean: 3.58, Std Dev: 1.89
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [49.0, 233.0, 13.0]
    
    Mean: 3.77, Std Dev: 1.93
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [53.0, 249.0, 14.0]
    
    Mean: 3.79, Std Dev: 1.86
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [56.0, 258.0, 15.0]
    
    Mean: 3.73, Std Dev: 1.81
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [61.0, 283.0, 16.0]
    
    Mean: 3.81, Std Dev: 1.78
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [63.0, 287.0, 17.0]
    
    Mean: 3.71, Std Dev: 1.77
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [66.0, 296.0, 18.0]
    
    Mean: 3.67, Std Dev: 1.73
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [71.0, 321.0, 19.0]
    
    Mean: 3.74, Std Dev: 1.71
    =============================
    Incoming Item: 1
    [LS,    SS,    N]
    [72.0, 322.0, 20.0]
    
    Mean: 3.60, Std Dev: 1.77
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [76.0, 338.0, 21.0]
    
    Mean: 3.62, Std Dev: 1.73
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [80.0, 354.0, 22.0]
    
    Mean: 3.64, Std Dev: 1.69
    =============================
    Incoming Item: 9
    [LS,    SS,    N]
    [89.0, 435.0, 23.0]
    
    Mean: 3.87, Std Dev: 1.98
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [94.0, 460.0, 24.0]
    
    Mean: 3.92, Std Dev: 1.96
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [98.0, 476.0, 25.0]
    
    Mean: 3.92, Std Dev: 1.92
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [101.0, 485.0, 26.0]
    
    Mean: 3.88, Std Dev: 1.89
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [104.0, 494.0, 27.0]
    
    Mean: 3.85, Std Dev: 1.86
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [109.0, 519.0, 28.0]
    
    Mean: 3.89, Std Dev: 1.84
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [111.0, 523.0, 29.0]
    
    Mean: 3.83, Std Dev: 1.84
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [115.0, 539.0, 30.0]
    
    Mean: 3.83, Std Dev: 1.81
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [118.0, 548.0, 31.0]
    
    Mean: 3.81, Std Dev: 1.79
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [124.0, 584.0, 32.0]
    
    Mean: 3.88, Std Dev: 1.80
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [129.0, 609.0, 33.0]
    
    Mean: 3.91, Std Dev: 1.78
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [131.0, 613.0, 34.0]
    
    Mean: 3.85, Std Dev: 1.78
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [137.0, 649.0, 35.0]
    
    Mean: 3.91, Std Dev: 1.79
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [139.0, 653.0, 36.0]
    
    Mean: 3.86, Std Dev: 1.80
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [143.0, 669.0, 37.0]
    
    Mean: 3.86, Std Dev: 1.77
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [148.0, 694.0, 38.0]
    
    Mean: 3.89, Std Dev: 1.76
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [153.0, 719.0, 39.0]
    
    Mean: 3.92, Std Dev: 1.75
    =============================
    Incoming Item: 1
    [LS,    SS,    N]
    [154.0, 720.0, 40.0]
    
    Mean: 3.85, Std Dev: 1.78
    =============================
    Incoming Item: 5
    [LS,    SS,    N]
    [159.0, 745.0, 41.0]
    
    Mean: 3.88, Std Dev: 1.77
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [163.0, 761.0, 42.0]
    
    Mean: 3.88, Std Dev: 1.75
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [167.0, 777.0, 43.0]
    
    Mean: 3.88, Std Dev: 1.73
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [169.0, 781.0, 44.0]
    
    Mean: 3.84, Std Dev: 1.73
    =============================
    Incoming Item: 7
    [LS,    SS,    N]
    [176.0, 830.0, 45.0]
    
    Mean: 3.91, Std Dev: 1.77
    =============================
    Incoming Item: 1
    [LS,    SS,    N]
    [177.0, 831.0, 46.0]
    
    Mean: 3.85, Std Dev: 1.81
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [180.0, 840.0, 47.0]
    
    Mean: 3.83, Std Dev: 1.79
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [183.0, 849.0, 48.0]
    
    Mean: 3.81, Std Dev: 1.78
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [187.0, 865.0, 49.0]
    
    Mean: 3.82, Std Dev: 1.76
    =============================
    Incoming Item: 7
    [LS,    SS,    N]
    [194.0, 914.0, 50.0]
    
    Mean: 3.88, Std Dev: 1.80
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [197.0, 923.0, 51.0]
    
    Mean: 3.86, Std Dev: 1.78
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [201.0, 939.0, 52.0]
    
    Mean: 3.87, Std Dev: 1.77
    =============================
    Incoming Item: 4
    [LS,    SS,    N]
    [205.0, 955.0, 53.0]
    
    Mean: 3.87, Std Dev: 1.75
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [211.0, 991.0, 54.0]
    
    Mean: 3.91, Std Dev: 1.76
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [217.0, 1027.0, 55.0]
    
    Mean: 3.95, Std Dev: 1.76
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [220.0, 1036.0, 56.0]
    
    Mean: 3.93, Std Dev: 1.75
    =============================
    Incoming Item: 3
    [LS,    SS,    N]
    [223.0, 1045.0, 57.0]
    
    Mean: 3.91, Std Dev: 1.74
    =============================
    Incoming Item: 2
    [LS,    SS,    N]
    [225.0, 1049.0, 58.0]
    
    Mean: 3.88, Std Dev: 1.74
    =============================
    Incoming Item: 6
    [LS,    SS,    N]
    [231.0, 1085.0, 59.0]
    
    Mean: 3.92, Std Dev: 1.75
    =============================
    Incoming Item: 1
    [LS,    SS,    N]
    [232.0, 1086.0, 60.0]
    
    Mean: 3.87, Std Dev: 1.77
    =============================
    
