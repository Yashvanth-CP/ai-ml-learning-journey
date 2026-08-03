import numpy as np
# to creat the array
# array = np.array([1,3,4,5,]) * 2

# # print(array)
# # print(type(array))

# # to te dimendion of the array
# # array2 =  np.array([[[1,2,3,4,5,], [234,35,64,65,343],[34,56,76,56,56]],
# #             [[1,2,3,4,5,], [234,35,64,65,343],[34,56,76,56,56]],
# #                 [[1,2,3,4,5,], [234,35,64,65,343],[34,56,76,56,56]] ])

# # print(array2.ndim)
# # print(array2.shape)
# # print(array2[0][2][4])
# # print(array2[0:2:1])

# # slicing pf array
# array3 =  np.array([[1,2,3,4,5,], [234,35,64,65,343],[34,56,76,56,56]])


# print(array3[0:3:2])

# print(array3[::-1])


# #to acesss column

# print(array3[:,0:4])

# array = np.array([[1,2,3,4]])

# aray1 = np.array([[1],[2],[3],[4]])
# # print(array.shape)
# # print(aray1.shape)
# # print(array * aray1 ** 3)

# print(np.sum(array))

rng = np.random.default_rng()

print(rng.integers(low= 1, high= 101, size= 4))