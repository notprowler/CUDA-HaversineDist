import sys
import numpy as np
sys.path.append('./build')

import pandas as pd
#import cudf
import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

#taxi = cudf.read_parquet("yellow_tripdata_2009-01.parquet")
taxi = pd.read_parquet("/tmp/tlcdata/yellow_tripdata_2009-01.parquet")

x1=taxi['Start_Lon'].to_numpy()
y1=taxi['Start_Lat'].to_numpy()
x2=taxi['End_Lon'].to_numpy()
y2=taxi['End_Lat'].to_numpy()
size=len(x1)
dist=np.zeros(size)
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)