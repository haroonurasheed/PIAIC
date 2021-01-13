#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


v = np.zeros(10)
print(v)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


v2= np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[4]:


shape_of_v2 = v2.shape


# 5. Print the type of the previous array in question 3

# In[5]:


print(shape_of_v2)


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[7]:


print(v2.shape)


# 8. Create a boolean array with all the True values

# In[8]:


bolVal = np.ones((10),dtype=bool)
print(bolVal)


# 9. Create a two dimensional array
# 
# 
# 

# In[9]:


a = [[1,2,3,4],[5,6,7,8]]
print(a)


# 10. Create a three dimensional array
# 
# 

# In[10]:


b = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
print(b)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[11]:


x = np.arange(1,10)
print(x)
x = x[::-1]
print(x)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[12]:


b = np.array([int(b==5) for b in range(10)])
print(b)


# 13. Create a 3x3 identity matrix

# In[13]:


arr3D = np.identity(3)
print(arr3D)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[14]:


arr = np.array([1, 2, 3, 4, 5])
arr.astype('float64')


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[15]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])

out_arr = np.multiply(arr1,arr2) 


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[16]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
comparison = arr1 == arr2 
equal_arrays = comparison.all() 
  
print(equal_arrays)


# 17. Extract all odd numbers from arr with values(0-9)

# In[17]:



arr=np.array([0,1,2,3,4,5,6,7,8,9])
odd=[n for n in arr if n%2]
print(odd)


# 18. Replace all odd numbers to -1 from previous array

# In[18]:


arr=np.array([1, 3, 5, 7, 9])
arr[arr%2==1]=-1
print(arr)


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[19]:


arr = np.arange(10)
arr[5] = 12
arr[6] = 12
arr[7] = 12
arr[8] = 12
print(arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[20]:


x = np.ones((5,5))
x[1:-1,1:-1] = 0
print(x)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[21]:


arr2d = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
np.where(arr2d==5,12,arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[22]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,0] = 64
print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[23]:


ar2d = np.arange(9).reshape(3,3)
print(ar2d[0])


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[24]:


print(ar2d[1])


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[25]:


print(ar2d[2])


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[26]:


from numpy import random
abc = random.randint(100, size = (10, 10))
print("array 10 by 10")
print(abc)
mini_value = np.amin(abc)
maxi_value = np.amax(abc)
print(f"Minimum value in above created array is {mini_value}")
print(f"Maximum value in above created array is {maxi_value}")


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[27]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(np.intersect1d(a, b))


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[28]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
print(f"Positions of common values in above two arrays {np.arange(len(a))[a==b]}")


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[29]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data[names != 'Will'])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[30]:


mask = (names != 'Joe') & (names != 'Will')
print(data[mask])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[31]:


b_arr = np.random.uniform(5,10, size=(5,3))
print(b_arr)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[32]:


arr=np.random.randint(low=1,high=16,size=(2,2,4) ,dtype=int)
arr


# 33. Swap axes of the array you created in Question 32

# In[33]:


arr.swapaxes(1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[34]:


bca = np.array([0.2, 49, 9, 4, 144, 36, 64, 121, 169, 100])
print(bca)
acb = np.sqrt(bca)
print(np.where(acb < 0.5, 1, acb))


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[35]:


a=np.random.random(12)
b=np.random.random(12)
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[36]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names=np.unique(names)
names.sort()
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[37]:


a=np.array([1,2,3,4,5])
b=np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[38]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 
newColumn = np.array([[10,10,10]])
sampleArray[1]=newColumn
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[39]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[40]:


matrix=np.random.randn(5,4)
csum=matrix.cumsum()
print(csum)

