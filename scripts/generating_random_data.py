import numpy as np

np.random.seed(42)
data = (np.random.randint(0,100, size=200))
print(f"daily sales figures:\n",data)


#calculate mean manually

total = 0
for value in data:
    total += value
mean= total / len(data)
print(f" mean data:\n {mean}")

#Calculate median manually
n = 200
sorted_data = np.sort(data)
middle1 = data[n//2 - 1]
middle2 = data[n//2]
median = (middle1 + middle2)/2

print(f"median data:\n {median}")

#range value manually
largest_data = data[0]
smallest_data = data[0]

for value in data:
    if value > largest_data:
        largest_data = value
    if value < smallest_data:
        smallest_data =value

range_data = largest_data - smallest_data
print("maximum:", largest_data)
print("minimum:", smallest_data)
print("range:",  range_data)


#standard deviation manually

total =0
for value in data:
    total += value
mean = total / len(data)
variance_sum = 0
for value in data:
    variance_sum += (value - mean)**2
variance = variance_sum / len(data)
std = variance**0.5
print(f"the standard deviation is :",std)

#the 25th percentile

n = 200
position = 0.25*(n-1)
lower = int(position)
upper = lower + 1
fraction = position - lower
q1 = data[lower] + fraction * (data[upper]- data[lower])
print(f"the 25th percentile of data is:",q1)

# calculatin the 75th percentile


q3_position  = 0.75*(n - 1)
q3  = data[int(q3_position)]
lower = int(q3_position)
upper = lower + 1
fraction = q3_position - lower
q3 = data[lower] + fraction *(data[upper] - data[lower])
print(f"the 75th percentile is :", q3)

iqr = q3 - q1
print(f"the iqr value is: " ,iqr)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 *iqr
print(f"the outliers boundary are :\nlower bound",lower_bound)
print(f"upper bound:",upper_bound)

# the outliers

outliers = []
for value in data:
    if value < lower_bound or value > upper_bound:
     outliers.append(value)
print(f"outliers:", outliers )


#normalisation
largest = data[0]
smallest = data[0]
for value in data:
    if value  > largest:
        largest = value
    if value < smallest:
        smallest = value

normalized = (data - data.min()) / (data.max() - data.min())
print(normalized [:10] )


#report section
#the mean value is 49.01, and the standard deviation value us 29.5, the standard deviation tells how the sales figures tend to vary from the average
#the standard deviation of 29.5 tells that sales are spread out

#reshaping data
sales_region = data.reshape(4,50)
print(f"4 sales region over 50 days:\n" ,sales_region)

region_mean = []
for rows in sales_region:
    total = 0
    for val in rows:
      total  += val
    mean = total / len(rows)
    region_mean.append(mean)
print(f"region mean :\n", region_mean)

highest_mean = region_mean[0]
best_region = 0
for i in range(len(region_mean)):
    if region_mean[i] > highest_mean:
        best_region = i
print("region means:",region_mean)
print("highest region:", best_region + 1)

