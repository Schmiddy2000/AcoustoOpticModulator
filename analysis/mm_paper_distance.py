# Imports

import numpy as np


"""
Idea:
Take the image with the largest frequency and most (prominent) dots created by higher order diffraction. 
Alternatively even use multiple images. 

Here we want to use the center of the 0th order dot as the center of the coordinate system. 
The other centers should then be assessed relative to this point. 
We can then do a linear regression, or rather an ODR, since we expect to have fairly large errors in x and y direction. 
We can then use the parameters obtained to write a function that can turn the measurements measured with neglection 
of the tilt into corrected values with corresponding uncertainties.  
"""

d_values = np.array([])

d = 47.5
h = 2.5

print(4200 * (np.sqrt(d**2 + h**2) / d - 1))
