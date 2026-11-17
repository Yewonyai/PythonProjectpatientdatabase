import pandas as pd


import numpy as np

arr = np.array([1,2,3,4,5],ndmin = 5)
print(arr)
print("the dimensions of the arrays are:", arr.ndim)

new_array = np.array([[1,2,3,5,56,6,7,7,777,88],
                      [33,444,55,6,3,6,73,2,4,88]],ndmin= 10)
print(new_array)
new_arrays =new_array.reshape(2,10)
print(new_arrays)

bilkisu = np.array([1,2,3,4,5,6,7,8,9,10])
x = bilkisu.copy()
x[0] = 20

bilkisu[0] =  42
print(bilkisu)
print(x)

# x =new_array.copy()
# new_array[0,0]=42
# print(x)
# # data = {"name":["Thor", "Hawkeye", "Bob","Captain america"],
#         "age":["40","37", "50", "36"]}
# df1 = pd.DataFrame(data)
# print(df1)
# print(df1.loc[2],df1.iloc[1])
# data2 = {"name":["Rock", "Batman", "Catwoman","Shazam"],
#         "age":["50","30", "27", "40"]}
# df2 = pd.DataFrame(data2)
# print(df2)
# df = pd.concat([df1,df2])
# df["age"] = df["age"].astype(int)
# print(df)
# print(df["name"])
# print(df["age"])
# print(df["age"] >= 40)
# print(df.age[df["age"] >=40])