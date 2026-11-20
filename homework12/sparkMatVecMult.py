import sys
from time import perf_counter
import numpy as np
from pyspark.sql import SparkSession

rows, cols, partitions = map(int, sys.argv[1:4])

#create spark session and context
spark = SparkSession\
        .builder\
        .appName("SparkMatVec")\
        .getOrCreate()

# get spark session, which is used to call spark methods
sc = spark.sparkContext

# Generate Matrix:
rng = np.random.default_rng(2026)
A = rng.random((rows,cols))
# generate vector:
x = rng.random(cols)

start = perf_counter()
# Create RDD of matrix A 
Aspark = sc.parallelize(A, partitions)

# compute b 
b = Aspark.map(lambda row: np.dot(row, x))

#Extract result from b RDD as numpy array
result = b.collect()
elapsed = perf_counter() - start
print(f"Rows: {rows}, columns: {cols}, partitions: {partitions}")
print(f"Time taken: {elapsed:.6f} seconds")

#check that the results match
if "--check" in sys.argv:
    desired = np.matmul(A, x)
    print("Matrix A:\n", A)
    print("Vector x:", x)
    print("Spark result b:", np.array(result))
    print("NumPy result:", desired)
    np.testing.assert_allclose(result, desired)
    print("Correctness check passed")

#stop Spark session
spark.stop()
