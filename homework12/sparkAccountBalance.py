
import sys
import re
from pyspark.sql import SparkSession

year = sys.argv[1]

# Create Spark session and context
spark = SparkSession\
        .builder\
        .appName("SparkAccountBalance")\
        .getOrCreate()

sc = spark.sparkContext

# Read CSV file as RDD
input_file = "Unclaimed_bank_accounts.csv"

accounts = spark.read.csv(
    input_file, header=True, inferSchema=True
).rdd.map(lambda x: tuple(x))

# Print the first 3 records
print(accounts.take(3))

# Remove records with missing balances
accounts = accounts.filter(lambda x: x[2] is not None)


# ==========================================
# Task 1: Average balance for each city
# ==========================================

def calculate_average(data):
    key, values = data
    average = sum(values) / len(values)
    return (key, average)


city_average = accounts\
    .map(lambda x: (x[4], x[2]))\
    .groupByKey()\
    .map(calculate_average)

print("\n=== Task 1: Average Balance by City ===")

for result in city_average.collect():
    print(result)


# ==========================================
# Task 2: Minimum and maximum balance
# ==========================================

def calculate_min_max(data):
    key, values = data
    minimum = min(values)
    maximum = max(values)
    return (key, minimum, maximum)


city_min_max = accounts\
    .map(lambda x: (x[4], x[2]))\
    .groupByKey()\
    .map(calculate_min_max)

print("\n=== Task 2: Min and Max Balance by City ===")

for result in city_min_max.collect():
    print(result)


# ==========================================
# Task 3: Top 3 and bottom 3 balances
# ==========================================

def calculate_top_bottom3(data):
    key, values = data
    balances = sorted(values)
    top3 = balances[-3:][::-1]
    bottom3 = balances[:3]
    return (key, top3, bottom3)


city_top_bottom3 = accounts\
    .map(lambda x: (x[4], x[2]))\
    .groupByKey()\
    .map(calculate_top_bottom3)

print("\n=== Task 3: Top 3 and Bottom 3 by City ===")

for result in city_top_bottom3.collect():
    print(result)


# ==========================================
# Task 4: Statistics for each city and bank
# ==========================================

city_bank_grouped = accounts\
    .map(lambda x: ((x[4], x[6]), x[2]))\
    .groupByKey()

# Average
city_bank_average = city_bank_grouped.map(calculate_average)

print("\n=== Task 4: Average by City and Bank ===")

for result in city_bank_average.collect():
    print(result)

# Minimum and maximum
city_bank_min_max = city_bank_grouped.map(calculate_min_max)

print("\n=== Task 4: Min and Max by City and Bank ===")

for result in city_bank_min_max.collect():
    print(result)

# Top 3 and bottom 3
city_bank_top_bottom3 = city_bank_grouped.map(
    calculate_top_bottom3
)

print("\n=== Task 4: Top 3 and Bottom 3 by City and Bank ===")

for result in city_bank_top_bottom3.collect():
    print(result)


# ==========================================
# Task 5: Filter accounts by transaction year
# ==========================================

filtered_accounts = accounts.filter(
    lambda x: re.search(
        rf"(?<!\d){year}(?!\d)", str(x[5])
    ) is not None
)

# Group filtered accounts by city
filtered_city_grouped = filtered_accounts\
    .map(lambda x: (x[4], x[2]))\
    .groupByKey()

# Average
filtered_city_average = filtered_city_grouped.map(
    calculate_average
)

print(f"\n=== Task 5: Average by City in {year} ===")

for result in filtered_city_average.collect():
    print(result)

# Minimum and maximum
filtered_city_min_max = filtered_city_grouped.map(
    calculate_min_max
)

print(f"\n=== Task 5: Min and Max by City in {year} ===")

for result in filtered_city_min_max.collect():
    print(result)

# Top 3 and bottom 3
filtered_city_top_bottom3 = filtered_city_grouped.map(
    calculate_top_bottom3
)

print(f"\n=== Task 5: Top 3 and Bottom 3 by City in {year} ===")

for result in filtered_city_top_bottom3.collect():
    print(result)


# Group filtered accounts by city and bank
filtered_city_bank_grouped = filtered_accounts\
    .map(lambda x: ((x[4], x[6]), x[2]))\
    .groupByKey()

# Average
filtered_city_bank_average = filtered_city_bank_grouped.map(
    calculate_average
)

print(f"\n=== Task 5: Average by City and Bank in {year} ===")

for result in filtered_city_bank_average.collect():
    print(result)

# Minimum and maximum
filtered_city_bank_min_max = filtered_city_bank_grouped.map(
    calculate_min_max
)

print(f"\n=== Task 5: Min and Max by City and Bank in {year} ===")

for result in filtered_city_bank_min_max.collect():
    print(result)

# Top 3 and bottom 3
filtered_city_bank_top_bottom3 = filtered_city_bank_grouped.map(
    calculate_top_bottom3
)

print(f"\n=== Task 5: Top 3 and Bottom 3 by City and Bank in {year} ===")

for result in filtered_city_bank_top_bottom3.collect():
    print(result)


# Stop Spark session
spark.stop()
