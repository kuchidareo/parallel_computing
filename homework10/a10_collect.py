import csv
import json
import os
import subprocess
import sys

env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = "1"
env["OMP_NUM_THREADS"] = "1"
process_counts = [1, 4, 9, 16]
N = 2304
n = 144

for experiment in ["strong", "weak"]:
    with open(f"a10_{experiment}.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["N", "P", "trial", "total"])
        writer.writeheader()
        for p in process_counts:
            matrix_size = N if experiment == "strong" else n*p
            print(f"{experiment}: N={matrix_size}, P={p}", flush=True)
            result = subprocess.run(
                ["mpirun", "-n", str(p), sys.executable,
                 "a10_benchmark.py", str(matrix_size)],
                env=env, capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(result.stdout)
                print(result.stderr)
                result.check_returncode()
            writer.writerows(json.loads(line) for line in result.stdout.splitlines())
            file.flush()
