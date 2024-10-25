import time
import numpy as np
from prefect import flow, task
import subprocess

@task()
def nvidia_smi_taks():
    output = subprocess.check_output(["nvidia-smi"])
    output = output.decode()
    for out in output.split("\n"):
        print(out)
    time.sleep(2)

@task()
def mult_task(num_operations: int = 1000, n: int = 100, m: int = 512):
    total_time = 0
    for _ in range(num_operations):
        mat = np.random.uniform(-1, 1, (n, m))

        st = time.perf_counter_ns()
        res = mat@mat.T
        end = time.perf_counter_ns()
        total_time += (end-st)
    print(f"Performed {num_operations} [{n}x{m}] matrix operaitons in {total_time // num_operations} ns.")
    

@flow(log_prints=True)
def dummy(num_operations: int = 1000, n: int = 100, m: int = 512):
    """Performs a matrix multiplication a specific number of times.

    Args:
        num_operations (int, optional): The number of operations to be computed. Defaults to 1000.
        n (int, optional): Number of columns in the matrix. Defaults to 100.
        m (int, optional): Number of rows in the matrix. Defaults to 512.
    """
    nvidia_smi_taks()
    mult_task(num_operations, n, m)

