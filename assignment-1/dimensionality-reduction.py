import os

# Summary Statistics:
#	              Min  Max   Mean    SD   Class Correlation
#   sepal length: 4.3  7.9   5.84  0.83    0.7826
#    sepal width: 2.0  4.4   3.05  0.43   -0.4194
#   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
#    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

sample_size = 150

sepal_length_mean = 5.84
sepal_width_mean = 3.05
petal_length_mean = 3.76
petal_width_mean = 1.20


fileDir = os.path.dirname(os.path.relpath('__file__'))
filename = os.path.join(fileDir, 'iris-data-set/iris.data')

data_file = open(filename, 'r')
results = [0 for i in range(4)]

for line in data_file.readlines():
    line = line.split(",")
    line.pop()

    results[0] += (float(line[0]) - sepal_length_mean) ** 2
    results[1] += (float(line[1]) - sepal_width_mean) ** 2
    results[2] += (float(line[2]) - petal_length_mean) ** 2
    results[3] += (float(line[3]) - petal_width_mean) ** 2

results = [round(result/sample_size,4) for result in results]

print(results)
print("0.6889, 0.1849, 3.0976, 0.5776")

data_file.close()
