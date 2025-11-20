import sys
sys.path.append('./build')

import haversine_library
import numpy as np

#"New York", "Paris", "Sydney", respectively 
x = np.array([-74.0060,2.3522,151.2093]);
y = np.array([40.7128,48.8566,-33.8688]);

assert (len(x)==len(y));

size=len(x)*len(x);
print("size=",size);
x1 = np.array([x[0],x[0],x[0],x[1],x[1],x[1],x[2],x[2],x[2]]);
y1 = np.array([y[0],y[0],y[0],y[1],y[1],y[1],y[2],y[2],y[2]]);
x2 = np.array([x[0],x[1],x[2],x[0],x[1],x[2],x[0],x[1],x[2]]);
y2 = np.array([y[0],y[1],y[2],y[0],y[1],y[2],y[0],y[1],y[2]]);
dist = np.zeros(size);

print("x1: ", x1)
print("y1: ", y1)
print("x2: ", x2)
print("y2: ", y2)

haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)

print("dist: ", dist)

assert np.allclose(dist,[
                0.0,
                5.83724090e03,
                1.59887555e04,
                5.83724090e03,
                0.0,
                1.69604974e04,
                1.59887555e04,
                1.69604974e04,
                0.0,
            ]
    )
