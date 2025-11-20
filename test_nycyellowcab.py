import sys
import numpy as np
import time
import math

# Ensure headless mode (prevents GDK display errors)
import matplotlib
matplotlib.use("Agg")

sys.path.append('./build')
import haversine_library
import matplotlib.pyplot as plt

# Use POLARS instead of Pandas -> MUCH faster for huge Parquet files
import polars as pl

DATA_PATH = "/tmp/tlcdata"

def combineAll():
    # Load 12 parquet files lazily (super fast)
    # Assumes files are named yellow_tripdata_2009-01.parquet, etc.
    files = [f"{DATA_PATH}/yellow_tripdata_2009-{i:02d}.parquet" for i in range(1,13)]
    
    try:
        df = pl.read_parquet(files)
    except Exception as e:
        print(f"Error loading parquet files from {DATA_PATH}. Ensure data exists.")
        raise e

    # Clean columns in one vectorized pass
    df = df.drop_nulls(["Start_Lon","Start_Lat","End_Lon","End_Lat"])
    df = df.filter(
        (df["Start_Lon"] != 0) &
        (df["Start_Lat"] != 0) &
        (df["End_Lon"]   != 0) &
        (df["End_Lat"]   != 0)
    )

    # NYC bounding box
    min_lon, min_lat = -74.15, 40.5774
    max_lon, max_lat = -73.7004, 40.9176

    df = df.filter(
        (df["Start_Lon"] >= min_lon) & (df["Start_Lon"] <= max_lon) &
        (df["Start_Lat"] >= min_lat) & (df["Start_Lat"] <= max_lat) &
        (df["End_Lon"]   >= min_lon) & (df["End_Lon"]   <= max_lon) &
        (df["End_Lat"]   >= min_lat) & (df["End_Lat"]   <= max_lat)
    )

    return df

def to_numpy(taxi):
    x1 = taxi["Start_Lon"].to_numpy()
    y1 = taxi["Start_Lat"].to_numpy()
    x2 = taxi["End_Lon"].to_numpy()
    y2 = taxi["End_Lat"].to_numpy()
    return x1, y1, x2, y2


def haversine_python(size, x1, y1, x2, y2, dist):
    R = 6371.0
    for i in range(size):
        lat1 = math.radians(y1[i])
        lon1 = math.radians(x1[i])
        lat2 = math.radians(y2[i])
        lon2 = math.radians(x2[i])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        dist[i] = R * c
    return dist

if __name__ == "__main__":
    print(f"Loading NYC Taxi data from {DATA_PATH} using Polars...")
    start_load = time.time()
    try:
        df = combineAll()
    except Exception as e:
        print(e)
        sys.exit(1)
        
    print(f"Data loaded in {time.time() - start_load:.2f} seconds.")
    print(f"Total rows: {len(df)}")

    x1, y1, x2, y2 = to_numpy(df)
    size = len(x1)
    
    # Prepare output arrays
    dist_gpu = np.zeros(size, dtype=np.float64)
    
    print("\n--- Running GPU Haversine ---")
    # The C++ wrapper prints its own timing for allocation/transfer/kernel
    try:
        haversine_library.haversine_distance(size, x1, y1, x2, y2, dist_gpu)
    except Exception as e:
        print(f"GPU Execution failed: {e}")
        sys.exit(1)

    print(f"GPU Calculated Sample: {dist_gpu[:5]}")

    # Validation
    # Since Python loops are extremely slow, we only validate the first N rows
    VALIDATION_SIZE = 100
    print(f"\n--- Validating on first {VALIDATION_SIZE} rows using CPU (Python Loop) ---")
    
    dist_cpu = np.zeros(VALIDATION_SIZE, dtype=np.float64)
    
    # Slice input for CPU
    x1_sub = x1[:VALIDATION_SIZE]
    y1_sub = y1[:VALIDATION_SIZE]
    x2_sub = x2[:VALIDATION_SIZE]
    y2_sub = y2[:VALIDATION_SIZE]

    start_cpu = time.time()
    haversine_python(VALIDATION_SIZE, x1_sub, y1_sub, x2_sub, y2_sub, dist_cpu)
    cpu_time = time.time() - start_cpu
    print(f"CPU time for {VALIDATION_SIZE} rows: {cpu_time:.4f}s")

    # Compare
    is_close = np.allclose(dist_gpu[:VALIDATION_SIZE], dist_cpu, atol=1e-2)
    print(f"Match Status: {'SUCCESS' if is_close else 'FAILED'}")
    
    if not is_close:
        print("Sample GPU:", dist_gpu[:5])
        print("Sample CPU:", dist_cpu[:5])
    
    print("\nScript Completed.")
