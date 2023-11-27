 # importing the modules 
import random
from matplotlib.pyplot import sci 
import numpy as np 
import matplotlib.pyplot as plt 
import math 
from scipy import stats as stats
import statistics 

# limits of integration 
a = -np.pi/2
b =  np.pi/2
exactValue = 0
NumAttempts = 5000  
# function to calculate the sin of a particular  
# value of x 
goldenValue = 2.0
def f(x): 
    return np.cos(x) 
    #return 5 + np.sin(x)    
"""  

# limits of integration 
a =  0
b =  1
exactValue = 0
NumAttempts = 5000
# function to calculate the sin of a particular  
# value of x 
goldenValue = 1.0206523024533902
def f(x):
    return 1 + x * np.cos(71*x) + np.sin(13*x)
"""

# list to store all the values for plotting  
plt_vals = [] 
plt_num_of_eval=[]
plt_std_devs = list()
plt_std_devs_accurate = list()
max_point = 1000000

antithetic_0       = True
antithetic_classic = False
target_error = float(5) #%
rng = np.random.default_rng(12345)
for i in range(1, NumAttempts):         
  if (i % 1000==0) :
     print ("Trajectory: ", i)     
  mean   = 0.0
  sum_sq = 0
  sumOfFunc = 0
  sum_sq_anthi = 0
  
  fvals = list()
  I = (b-a)
  xprev = 0
  for pointIdx in range(1,max_point): 
      #x = random.uniform(a,b)  
      x = rng.uniform(a,b)
      if antithetic_0 and pointIdx%2 == 0:
         x = b - xprev + a
         
      fx = I*f(x)     
            
      if (i % 10000 == 0) :
        print ("\tPoint: ", i) 
        
      if antithetic_classic:
         x1 = b - xprev + a
         fx = fx + I*f(x1) 
         fx = fx/2
         
      sumOfFunc += fx   
      
      sum_sq   += fx*fx    
"""      
      if False and antithetic_0 and pointIdx%2 == 0:
         #try to emulate "extra" variant 
         #first remove what we shoujld not have added
         #now add what we wanted
         fa = ((fx + fxprev)*0.5) #? is that what extra anti wants
         sum_sq_anthi += fa*fa
         sum_sq = sum_sq_anthi 
 """     
      fxprev = fx
      xprev = x

      #fvals.append(fx)
      if (pointIdx > 10 and pointIdx % 1==0):
        mean =  sumOfFunc/pointIdx   
        variance_2 = (sum_sq /float(pointIdx) - mean*mean) /float(pointIdx-1)        
        
        eps = 1e-8;
        if (variance_2 < -eps) : 
           variance_2 = 0
           
        std_dev = math.sqrt(variance_2)
        error = 100 * std_dev / abs(mean)
        if(error < target_error):
            break

  std_dev_accurate = 0
  for ff in fvals:
    std_dev_accurate += (ff-mean)*(ff-mean)
  std_dev_accurate = 100 * std_dev_accurate/float(pointIdx-1) 
  std_dev_accurate = std_dev_accurate  / abs(mean) #relative
  
  plt_std_devs_accurate.append(std_dev_accurate)
  
  ans = sumOfFunc/float(pointIdx)  
  plt_vals.append(ans) 
  plt_num_of_eval.append(pointIdx)
  plt_std_devs.append(error)


std_dev = math.sqrt(np.var(plt_vals))
post_sigma = std_dev/goldenValue*100
print (" Golden function value: ", goldenValue, " average ", np.average(plt_vals), " std dev ", std_dev, " sigma ", post_sigma, "%", " vs target ", post_sigma/target_error)
print ("Average number of point to converge: ", np.average(plt_num_of_eval))
#print 
print (stats.normaltest(plt_vals))

"""     
  #figure, axis = plt.subplots(2, 2) plt.title("Distributions of areas calculated") 
plt.hist (plt_vals, bins=100, ec="black")  
plt.xlabel("Integral value") 
plt.hist (plt_vals, bins=100, ec="black")  
plt.show() # shows the plot
 
plt.title("Number of point for convergence") 
plt.hist (plt_num_of_eval, bins=50, ec="black")  
plt.show() # shows the plot
plt.clf()
"""     
real_error          = [100*(x - goldenValue)/goldenValue for x in plt_vals]

plt.title("Postfactum errors of integral in %, target is " + str(target_error)) 
plt.hist (real_error, bins=100, ec="black", density=True)  
x = np.arange(-5*target_error, 5 *target_error, 0.1)
plt.plot(x, stats.norm.pdf(x, 0, target_error), color='green')
plt.plot(x, stats.norm.pdf(x, 0, post_sigma), color='red')
plt.show() # shows the plot

#plt.plot (real_error,  color="black")  


plt.pause(50)



