---
title: Python数据处理教程
date: 2017-06-12 15:22:58
comments: false
tags:
- Python
---
本文主要介绍Python在数据处理中用到的库和方法。
<!--more-->
这篇文章本质上是一篇翻译文，原文是[Python Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)。

Python是一门优秀的通用编程语言，再结合一些流行的库（numpy，scipy，matplotlib）它会成为强大的科学计算环境。


# Python
python是一门高级的动态类型的多模式编程语言。Python代码经常被说成是伪代码，因为它可以让你用可读性很强的几行代码来实现一些有力的想法。下面是一个用Python实现的经典的快速排序算法。
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# Prints "[1, 1, 2, 3, 6, 8, 10]"
```
## Python版本
目前，Python有两个不同的受支持版本2.7和3.5。令人困惑的是Python3.0引入了许多向后不兼容的改变，因此2.7版本下的代码在3.5中不一定有效，反之亦然。本教程的代码使用的是Python3.5。
可以通过命令行运行`Python --version`来检查你的Python版本。

## 基本数据类型
像大多数编程语言一样，Python有许多基本的数据类型，包括整型、浮点型、布尔型和字符串。这些数据类型的操作方式和其他编程语言类似。
**数字：**整型和浮点型的工作方式和其他编程语言类似：
```Python
x = 3
print(type(x)) # Prints "<class 'int'>"
print(x)       # Prints "3"
print(x + 1)   # Addition; prints "4"
print(x - 1)   # Subtraction; prints "2"
print(x * 2)   # Multiplication; prints "6"
print(x ** 2)  # Exponentiation; prints "9"
x += 1
print(x)  # Prints "4"
x *= 2
print(x)  # Prints "8"
y = 2.5
print(type(y)) # Prints "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # Prints "2.5 3.5 5.0 6.25"
```
注意：与其他编程语言不同，Python没有自加(`x++`)和自减(`x--`)操作。
Python也有内建的复数类型，具体内容可以参考[官方文档](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex)
**布尔值：**Python可以通过英文单词来实现常用的布尔逻辑操作，而不是使用符号(`&&`,`||`,等):
```Python
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
```
**字符串：**Python可以很好地支持字符串操作：
```Python
hello = 'hello'    # String literals can use single quotes
world = "world"    # or double quotes; it does not matter.
print(hello)       # Prints "hello"
print(len(hello))  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print(hw)  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print(hw12)  # prints "hello world 12"
```
字符串对象有许多非常有用的方法，比如：
```Python
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```
你可以通过[官方文档](https://docs.python.org/3.5/library/stdtypes.html#string-methods)来查找所有的字符串方法。

## 容器
Python包含几个内建的容器类型：列表、字典、集合和元组。

### 列表
Python中列表等效于数组，并且可以调整大小以及包含其他类型的元素：
```Python
xs = [3, 1, 2]    # Create a list
print(xs, xs[2])  # Prints "[3, 1, 2] 2"
print(xs[-1])     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'     # Lists can contain elements of different types
print(xs)         # Prints "[3, 1, 'foo']"
xs.append('bar')  # Add a new element to the end of the list
print(xs)         # Prints "[3, 1, 'foo', 'bar']"
x = xs.pop()      # Remove and return the last element of the list
print(x, xs)      # Prints "bar [3, 1, 'foo']"
```
你也可以在[官方文档](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists)中查找列表的详细资料。
**切片：**除了每次可以获取列表中的元素之外，Python还提供了简明的语法来获取子列表。这就是所谓的切片：
```Python
nums = list(range(5))     # range is a built-in function that creates a list of integers
print(nums)               # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])          # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])           # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])           # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])            # Get a slice of the whole list; prints "[0, 1, 2, 3, 4]"
print(nums[:-1])          # Slice indices can be negative; prints "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # Assign a new sublist to a slice
print(nums)               # Prints "[0, 1, 8, 9, 4]"
```
我们还将会在numpy的数部分作中遇到切片操作。
**循环：**你可以像下面的例子一样通过列表中的元素来实现循环：
```Python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# Prints "cat", "dog", "monkey", each on its own line.
```
如果想要得到每一个元素在循环体中的索引，可以使用内建函数`enumerate()`:
```Python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```
**列表解析：**在编程过程中，通常我们会将一种类型的数据转换为其他类型，考虑下面计算平方数的例子：
```Python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # Prints [0, 1, 4, 9, 16]
```
你可以利用简单地利用列表解析来实现：
```Python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # Prints [0, 1, 4, 9, 16]
```
列表解析可以包含条件：
```Python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # Prints "[0, 4, 16]"
```

### 字典
字典存储键值对(关键词，数值)，类似于Java中的`Map`和JavaScript中的对象。可以通过下面的例子来使用字典：
```Python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'     # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"
# print(d['monkey'])  # KeyError: 'monkey' not a key of d
print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"
del d['fish']         # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"
```
可以在[官方文档](https://docs.python.org/3.5/library/stdtypes.html#dict)中查找需要的有关字典的信息。
**循环：**可以简单地通过字典中的关键字来进行迭代：
```Python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```
如果想要得到字典中的关键字和相对应的数值，可以使用`items`方法来实现：
```Python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# Prints "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```
**字典解析：**与列表解析类似，可以方便地构造字典。例如：
```Python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```

### 集合
集合是不同元素的无序集合。思考下面的例子：
```Python
animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
animals.add('fish')       # Add an element to a set
print('fish' in animals)  # Prints "True"
print(len(animals))       # Number of elements in a set; prints "3"
animals.add('cat')        # Adding an element that is already in the set does nothing
print(len(animals))       # Prints "3"
animals.remove('cat')     # Remove an element from a set
print(len(animals))       # Prints "2"
```
同样可以在[官方文档](https://docs.python.org/3.5/library/stdtypes.html#set)中查找更详细的信息。
**循环：**通过集合迭代和通过列表迭代有相同的语法。但是由于集合是无序的，因此不能假设所访问的集合中元素的顺序：
```Python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# Prints "#1: fish", "#2: dog", "#3: cat"
```
**集合解析：**与列表解析和字典解析类似，我们可以使用集合解析快速地构造集合：
```Python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # Prints "{0, 1, 2, 3, 4, 5}"
```

### 元组
元组是(不可改变的)有序的值列表，元组在许多方面和列表类似，最重要的一点不同是元组可以作为字典中的关键字和集合中的元素，而列表不行。下面是一些小例子：
```Python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)        # Create a tuple
print(type(t))    # Prints "<class 'tuple'>"
print(d[t])       # Prints "5"
print(d[(1, 2)])  # Prints "1"
```
关于元组的[官方文档](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences)


## 函数
Python中的函数通过关键字`def`来定义。例如：
```Python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# Prints "negative", "zero", "positive"
```
通常会采用可选关键字参数来定义函数，例如：
```Python
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```
有关Python函数的更多信息可以参考[官方文档](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions)


## 类
Python中定义类的语法非常简洁：
```Python
class Greeter(object):

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```
Python中类的[官方文档](https://docs.python.org/3.5/tutorial/classes.html)

# Numpy
[Numpy](http://www.numpy.org/)是Python科学计算的核心库。它提供可高性能的多维数组对象以及处理多维数组的工具。如果你已经熟悉MATLAB，你会发现[教程](http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users)对你入门Numpy非常有用。

## 数组
Numpy中的数组是类型相同，由非负整数构成的元组索引的值的网格。维数是数组的秩(*rank*)，矩阵的形状(*shape*)是一个整数元组，表示每一个维度的大小。
可以通过嵌套Python列表的方式来初始化numpy数组，可以通过方括号来获取数组中的元素：
```Python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # Prints "<class 'numpy.ndarray'>"
print(a.shape)            # Prints "(3,)"
print(a[0], a[1], a[2])   # Prints "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # Prints "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # Prints "1 2 4"
```
Numpy也提供了很多函数用来创建数组：
```Python
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # Prints "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # Prints "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # Prints "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```
可以通过[官方文档](https://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation)来查找更多数组创建的方法。

## 数组索引
Numpy提供了几种数组索引的方法。
**切片：**类似于Python中的列表，numpy中的数组也可以切片。由于数组有时候是多维的，所以一定要明确数组中每一维度上的切片：
```Python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77"
```
可以混合使用整数索引和切片索引。然而，这样做需要数组的秩要低于原始数组。注意，这和MATLAB中数组切片的操作不同：
```Python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```
**整数数组索引：**当使用切片来索引numpy的数组时，结果被视为是原始阵列的一个子阵列。相反，整数数组索引可以利用其它数组中的数据来构造任意的数组。例如：
```Python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"
```
整数数组索引的一个有用的技巧是用用来选择或改变矩阵中每一列的元素
```Python
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```
**布尔数组索引：**布尔数组索引可以挑出数组中的任意元素。通常这用索引用来选择数组中满足一定条件的数组。例如：
```Python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"
```
想要了解更多关于numpy数组索引的信息，可以阅读[官方文档](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html)

## 数据类型
每一个numpy数组是一个有相同类型元素的网格。Numpy提供了许多用来构造数组的数据类型。当你在创建数组的时候Numpy会尝试猜测数组的类型，但是构造数组通常包括一个可选参数来明确数组的数据类型。例如：
```Python
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"
```
更多关于numpy的数据类型请参考[官方文档](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)

## 数据计算
在Numpy模块中基本的针对数组元素的数学公式操作可以通过运算符重载或函数来实现：
```Python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```
注意，不像在MATLAB中，`*`是元素间相乘，而不是矩阵相乘。我们使用`dot`函数来计算向量的内积，来实现矩阵乘法。`dot`可以通过模块函数或者数组对象的方法来实现：
```Python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```
Numpy在矩阵运算中提供了许多有用的函数，最常用的函数之一是`sum`:
```Python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```
可以在[官方文档](https://docs.scipy.org/doc/numpy/reference/routines.math.html)中查看完整的数学函数列表。
除了使用数组来计算数学函数之外，通常需要改变数组形状或者其他队数组中数据的操作，最简单的例子是矩阵的转置操作；通过数组对象的`T`转置可以简单地实现矩阵的转置。
```Python
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"
```
Numpy提供了许多函数来对数组进行操作，完整的函数列表请参考[官方文档](https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html)

## 广播
广播是一种非常强大的机制，可以允许numpy中不同形状的数组执行算术运算。通常情况下当有一个小数组和一个大数组时，想要用小数组在大数组上多次执行某些操作。
比如，想给矩阵的每一行加上常数向量。方法如下：
```Python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```
然而当矩阵`x`非常大的时候，Python中计算循环会变得很慢。注意到在矩阵`x`的每一行加上向量`v`等效于通过垂直复制`v`来构造矩阵`vv`，然后对`x`和`vv`执行元素间求和。实现方法如下：
```Python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```
Numpy广播允许执行这样的操作而不必创造多重复制的`v`，使用广播来实现：
```Python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1,0,1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```
即使`x`的形状是`(4,3)`而且`v`的形状是`(3,)`，但现行方程`y = x + v`由于广播机制同样有效。这个线性方程仿佛`v`的形状实际上是`(4,3)`，其中每一行都是`v`的复制，然后在执行元素间求和。
两个矩阵执行广播需要遵循以下规则：
1. 如果数组具有不同的秩，用1来填充秩较低的数组的形状，直到数组具有相同的长度。
2. 数组在这一维度中被称为兼容的，如果两个数组在某一维度中的大小的是相同的，或者其中一个数组的在某一维度中的大小是1。
3. 如果数组在所有维度上都是兼容的，则数组可以执行广播操作合并在一起。
4. 广播操作之后，数组的形状等于两个输入数组形状元素的最大值。
5. 在任何维度中，如果一个数组的形状为1并且另一个数组的形状大于1，第一个数组沿这个维度进行复制操作。

如果以上解释不明白的话，请参考[官方文档](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)的解释或者[其他解释](http://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc)
支持广播功能被称作通用功能，可以在[官方文档](https://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs)中查看所有通用功能。
下面是一些广播功能的应用：
```Python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```
广播功能可以使代码更加简洁和高效，因此可以尽可能地使用这用方法。

以上是Numpy中重要内容的简明概述，但实际远不止如此。可以在[numpy参考](https://docs.scipy.org/doc/numpy/reference/)中查看更多关于numpy的资料。

# SciPy
Numpy提供了高性能的多维数组的基本计算和操作工具，[Scipy](https://docs.scipy.org/doc/scipy/reference/)在这基础上，提供了大量用来操作numpy数组的函数，在不同的科学和工程领域都十分有用。
最有效的了解SciPy的方法是阅读[官方文档](https://docs.scipy.org/doc/scipy/reference/index.html)，这里会重点介绍SciPy中常用的部分。

## 图像操作
SciPy提供可一些图像操作的基本功能。例如，将磁盘中的图片读取到numpy数组中，将数组中的图像写入到磁盘中，以及改变图片的大小。下面是一些用来展示以上功能的小例子：
```Python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```
![原始图片](http://cs231n.github.io/assets/cat.jpg )
![修改颜色和大小之后的图片](http://cs231n.github.io/assets/cat_tinted.jpg)

## Matlab文件
函数`scipy.io.loadmat`和`scipy.io.savemat`可以对MATLAB文件进行读写，详细说明请参考[官方文档](https://docs.scipy.org/doc/scipy/reference/io.html)

## 两点间距离
Scipy定义了许多函数用来计算集合中坐标点之间的距离
函数`scipy.spatial.distance.pdist`可以计算集合中所有点之间的距离：
```Python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```
关于这个函数的详细资料可以阅读[官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)


# Matplotlib
[Matplotlib](http://matplotlib.org/)是一个绘图库，本节内容将简明介绍`matplotlib.pyplot`模块，该模块提供了与MATLAB类似的绘图系统。
类似的函数(`scipy.spatial.distance.cdist`)计算两个集合之间点的距离。[官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)

## 绘图
matplotlib中最重要的函数是`plot`，该函数可以绘制2D数据，例如“
```Python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```
运行上面的代码可以得到下面的图：
![](http://cs231n.github.io/assets/sine.png)
仅仅需要少量的额外工作就可以简单地绘制多条曲线，并且添加标题，图例和坐标轴
```Python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```
![](http://cs231n.github.io/assets/sine_cosine.png)
可以在[官方文档](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)中阅读关于`plot`的更多资料。

## 子图
使用`subplot`函数可以在一张图片上显示不同事物。例子如下：
```Python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```
![](http://cs231n.github.io/assets/sine_cosine_subplot.png)
有关`subplot`的更多信息请参考[官方文档](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot)

## 图像
可以使用`imshow`函数来显示图像，例如：
```Python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```
![](http://cs231n.github.io/assets/cat_tinted_imshow.png)