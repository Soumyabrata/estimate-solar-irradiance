import numpy as np
# Marks the indices that are clean (and not outliers)
def remove_outliers(data):
   #data = [2,4,5,1,6,5,40]
    
   u = np.mean(data)
   s = np.std(data)
   
   const = 3
    
   max_dev = u + const*s
   min_dev = u - const*s
    
   len_data = len(data)

   clean_index = []
   dirty_index = []
    
   for e in range(0,len_data):
      if min_dev<data[e] and data[e]<max_dev:
         clean_index.append(e)
      else:
         dirty_index.append(e)

   return (dirty_index,clean_index)
