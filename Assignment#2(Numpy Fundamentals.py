#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[3]:


import numpy as np
oneD=np.arange(10)
twoD=oneD.reshape(2,5)
twoD


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[6]:


a=np.arange(20).reshape(4,5)
a[2:4]=1
a


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[7]:


a=np.arange(20).reshape(2,10)
a[0][5:10]=1
a[1][5:10]=1
a[1][0]=5
a[1][1]=6
a[1][2]=7
a[1][3]=8
a[1][4]=9
a


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[21]:


arr=np.arange(10).reshape(2,5)
arr
arr.ravel()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[33]:


arr=np.arange(16).reshape(2,2,4)
arr.ravel()


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[46]:


arr=np.arange(15).reshape(1,5,3)
arr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[47]:


arr=np.random.rand(5,5)*100
print(np.square(arr))


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[54]:


arr=np.random.randint(2,9,size=(6,6))
np.mean(arr)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[59]:


arr=np.random.randint(2,9,size=(6,6))
np.std(arr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[58]:


arr=np.random.randint(2,9,size=(6,6))
np.median(arr)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[60]:


arr=np.random.randint(2,9,size=(6,6))
arr.transpose()


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[61]:


arr=np.random.randint(2,9,size=(6,6))
print(arr,np.trace(arr))


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[62]:


arr=np.random.randint(2,9,size=(6,6))
print(np.linalg.det(arr))


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[63]:


arr=np.arange(20)
print("5TH Percentile of Array :",np.percentile(arr,5))
print("95TH Percentile of Array :",np.percentile(arr,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[64]:


arr=np.identity(10)
arr[0,0]=None
print(np.isnan(arr))

