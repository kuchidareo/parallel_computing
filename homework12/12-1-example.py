import numpy as np
from pyspark.sql import SparkSession

#create spark session and context
spark = SparkSession\
        .builder\
        .appName("SparkMatVec")\
        .getOrCreate()

# get spark session, which is used to call spark methods
sc = spark.sparkContext

rows = 10 #number of rows
cols = 10 #number of columns
partitions = 2 #number of spark partitions

# Generate Matrix:
A = np.random.random((rows,cols))
# Create RDD of matrix A 
Aspark = sc.parallelize(A, partitions)

# generate vector: 
x = np.random.random(cols)

# compute b 
b = Aspark.map(lambda row: np.dot(row, x))

#Extract result from b RDD as numpy array
result = b.collect()

#check that the results match
desired = np.matmul(A, x)
same = np.allclose(result, desired)
print("A:", A)
print("x:", x)
print("result:", result)
print("desired:", desired)
print("same" if same else "not the same")
np.testing.assert_allclose(result, desired)

#stop Spark session
spark.stop()