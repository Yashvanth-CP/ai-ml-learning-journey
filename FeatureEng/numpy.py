import numpy as np
arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr)

#11
arr.shape

#12
print(arr.shape)
print(arr.ndim)
print(arr.size)
print(arr.dtype)
print(arr.itemsize)
print(arr.nbytes)
#int64 in the output means "64-bit signed integer data type"

#13
import numpy as np
arr=np.array([10,20,30,40,50])
arr[0]

#14 1D Indexing
import numpy as np
arr=np.array([10,20,30,40,50])
print(arr[0])
print(arr[2])
print(arr[-1])
print(arr[-2])

#15 2D Indexing
arr=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr[0,0])
print(arr[1,2])
print(arr[2,1])

#16 1D slicing
arr=np.array([10,20,30,40,50])
arr[1:4]

#17
arr=np.array([10,20,30,40,50])
print(arr[1:4])
print(arr[:3])
print(arr[2:])
print(arr[::2])


#18
arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr[0:2,1:3]

#19 step slicing
arr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(arr[0:2,1:3])
print("--------------")
print(arr[:,2])
print("--------------")
print(arr[1:,:])
print("--------------")
print(arr[::2,::2])

#20 Advanced Indexing
#Boolean indexing evaluates a conditional logic check over your target array, establishing a logical "mask". It extracts only the individual array items that align with a True value.


arr=np.array([10,15,20,25,30])
mask=arr>18
arr[mask]

#21 Fancy indexing
# fancy indexing allows us to use an array of indices to access multiple array elements at once
arr=np.array([10,20,30,40,50])
idx=[0,2,4]
arr[idx]

#22 Arithmetic Operations
import numpy as np
A=np.array([1,2,3])
B=np.array([4,5,6])
A+B
