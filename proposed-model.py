# Import the libraries

import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import operator
import bisect
import datetime
import pysolar
from pysolar.solar import *
from sklearn import linear_model, datasets
from scipy.interpolate import UnivariateSpline
from tempfile import TemporaryFile
from matplotlib.dates import DateFormatter

# User defined functions
from scripts.SG_solarmodel import *

# Collecting model files
model_loc = './JSTARfiles/ModelFiles/ModelFiles*.txt'
model_files = glob.glob(model_loc)
model_files = sorted(model_files)

fn_cat = []
etime_cat = []
iso_cat = []
lumi_cat = []
lumiNORM_cat = []
solar_cat = []
solarNORM_cat = []

for no_of_files, file_name3 in enumerate(model_files):

    # read the file
    with open(file_name3) as f:  # f is a file header
        reader = csv.reader(f, delimiter=",")
        d = list(reader)  # d is a list of list here.
        total_rows = len(d)

        # variable i starts from 1 so that header is skipped
        for i in range(1, total_rows):
            item1 = d[i][1]
            item2 = d[i][2]
            item3 = d[i][3]
            item4 = d[i][4]
            item5 = d[i][5]
            item6 = d[i][6]
            item7 = d[i][7]

            fn_cat.append(float(item1))
            etime_cat.append(float(item2))
            iso_cat.append(float(item3))
            lumi_cat.append(float(item4))
            lumiNORM_cat.append(float(item5))
            solar_cat.append(float(item6))
            solarNORM_cat.append(float(item7))

# Converting to numpys
fn_cat = np.array(fn_cat)
etime_cat = np.array(etime_cat)
iso_cat = np.array(iso_cat)
lumi_cat = np.array(lumi_cat)
lumiNORM_cat = np.array(lumiNORM_cat)
solar_cat = np.array(solar_cat)
solarNORM_cat = np.array(solarNORM_cat)

print(['The unique F-numbers of the camera are: ', np.unique(fn_cat)])


# Figure 15: Histogram of the distribution of F numbers
print('Generating figure for showing the distribution of F-number')
weights = (np.ones_like(fn_cat)/float(len(fn_cat)))*100
fig = plt.figure(1, figsize=(5, 4))
n, bins, patches = plt.hist(fn_cat, 20, facecolor='green', alpha=0.75, weights=weights)
plt.xlabel('F-number of captured images',fontsize=14)
plt.ylabel('Percentage of occurrence [%]',fontsize=14)
plt.grid(True)
fig.tight_layout()
plt.savefig('./results/Figure15.pdf', format='pdf')
plt.show()


# ========================================================================================
# For further analysis, we collect only the proper model files, where F-number is not relatively extremely large
considered_index = np.where( fn_cat < 40)
considered_index = np.array(considered_index)
considered_index = considered_index[0]
etime_taken = etime_cat[considered_index]


# Select the considered values. We need to perform this when it is a 1D feature
lumi_taken = lumi_cat[considered_index]
solar_taken = solar_cat[considered_index]
fn_taken = fn_cat[considered_index]
ip_vect = lumi_taken.reshape(-1, 1)
op_vect = solar_taken.reshape(-1, 1)

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# Figure: generating the linear model
plt.figure(2, figsize=(6, 5))
plt.scatter(ip_vect, op_vect, marker='o', linewidths=0.005, alpha=0.01)
plt.xlabel(r'Image luminance $\mathcal{L}$', fontsize=14)
plt.ylabel(r'Solar irradiance $\mathcal{S}$ [W/m$^2$]', fontsize=14)

# Fit line using all data
model = linear_model.LinearRegression()
model.fit(ip_vect, op_vect)
m0 = model.coef_
c0 = model.intercept_
m0 = m0[0][0]
c0 = c0[0]
print('Slope = ', m0, 'and Intercept = ', c0)


# Plot the linear regressor line
line_X = np.arange(3000, 70000)
line_y = model.predict(line_X[:, np.newaxis]) #Linear
plt.plot(line_X, line_y, '-r', label='Linear model')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./results/Figure7a.png', format='png')
plt.show()



# evaluation of the linear model
actual_solar_radiation = op_vect
estimated_solar_radiation_linear = [x * m0 for x in ip_vect] + c0
RMSE_linear = np.sqrt(np.mean((estimated_solar_radiation_linear-actual_solar_radiation)**2))
print('Root mean square for linear model = ', RMSE_linear, 'Watt/m2')



# -------------------
# second order polynomial
input_array = lumi_taken
output_array = solar_taken
z = np.polyfit(input_array, output_array, 2)
z0 = z[0]
z1 = z[1]
z2 = z[2]

# evaluation of the second order model
actual_solar_radiation = output_array
estimated_solar_radiation_quad = z0*pow(input_array, 2) + z1*pow(input_array, 1) + z2
RMSE_quadratic = np.sqrt(np.mean((estimated_solar_radiation_quad-actual_solar_radiation)**2))
print('Root mean square for quadratic model = ', RMSE_quadratic, 'Watt/m2')


plt.figure(3, figsize=(6, 5))
plt.scatter(ip_vect, op_vect, marker='o', linewidths=0.005, alpha=0.01)
plt.xlabel(r'Image luminance $\mathcal{L}$', fontsize=14)
plt.ylabel(r'Solar irradiance $\mathcal{S}$ [W/m$^2$]', fontsize=14)
line_X = np.arange(3000, 70000)
line_y = z0*pow(line_X, 2) + z1*pow(line_X, 1) + z2
plt.plot(line_X, line_y, '-r', label='Quadratic model')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./results/Figure7b.png', format='png')
plt.show()


# -------------------
# cubic order polynomial
input_array = lumi_taken
output_array = solar_taken
z = np.polyfit(input_array, output_array, 3)
z0 = z[0]
z1 = z[1]
z2 = z[2]
z3 = z[3]
print(z0, z1, z2, z3)

# evaluation of the cubic order model
actual_solar_radiation = output_array
estimated_solar_radiation_cube = z0*pow(input_array, 3) + z1*pow(input_array, 2) + z2*pow(input_array, 1) + z3
RMSE_cube = np.sqrt(np.mean((estimated_solar_radiation_cube-actual_solar_radiation)**2))
print('Root mean square for cubic model = ', RMSE_cube, 'Watt/m2')

plt.figure(4, figsize=(6, 5))
plt.scatter(ip_vect, op_vect, marker='o', linewidths=0.005, alpha=0.01)
plt.xlabel(r'Image luminance $\mathcal{L}$', fontsize=14)
plt.ylabel(r'Solar irradiance $\mathcal{S}$ [W/m$^2$]', fontsize=14)
line_X = np.arange(3000, 70000)
line_y = z0*pow(line_X, 3) + z1*pow(line_X, 2) + z2*pow(line_X, 1) + z3
plt.plot(line_X, line_y, '-r', label='Cubic model')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./results/Figure7c.png', format='png')
plt.show()


# More evaluation of the cubic model
diff_solar_L = actual_solar_radiation - estimated_solar_radiation_cube
weights = (np.ones_like(diff_solar_L)/float(len(diff_solar_L)))*100
plt.figure(5, figsize=(5, 4))
data = diff_solar_L
binwidth = 200
bins = np.arange(min(data), max(data) + binwidth, binwidth) - 20
n, opbins, patches =plt.hist(data, bins, facecolor='green', alpha=0.75, weights=weights)
plt.xlabel(r'Difference of solar irradiance [W/m$^2$]', fontsize=14)
plt.ylabel('Percentage of occurrence [%]',fontsize=14)
plt.grid(True)
plt.xlim((-800,800))
plt.savefig('./results/Figure9.pdf', format='pdf')


# -------------------
# fourth order polynomial
input_array = lumi_taken
output_array = solar_taken
z = np.polyfit(input_array, output_array, 4)
z0 = z[0]
z1 = z[1]
z2 = z[2]
z3 = z[3]
z4 = z[4]

# evaluation of the forth order model
actual_solar_radiation = output_array
estimated_solar_radiation_forth = z0*pow(input_array, 4) + z1*pow(input_array, 3) + z2*pow(input_array, 2) + z3*pow(input_array, 1) + z4
RMSE_forth = np.sqrt(np.mean((estimated_solar_radiation_forth-actual_solar_radiation)**2))
print('Root mean square for forth model = ', RMSE_forth, 'Watt/m2')


plt.figure(6, figsize=(6, 5))
plt.scatter(ip_vect, op_vect, marker='o', linewidths=0.005, alpha=0.01)
plt.xlabel(r'Image luminance $\mathcal{L}$', fontsize=14)
plt.ylabel(r'Solar irradiance $\mathcal{S}$ [W/m$^2$]', fontsize=14)
line_X = np.arange(3000, 70000)
line_y = z0*pow(line_X, 4) + z1*pow(line_X, 3) + z2*pow(line_X, 2) + z3*pow(line_X, 1) + z4
plt.plot(line_X, line_y, '-r', label='Quartic model')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./results/quartic-model.png', format='png')
plt.show()




# -------------------
# fifth order polynomial
input_array = lumi_taken
output_array = solar_taken
z = np.polyfit(input_array, output_array, 5)
z0 = z[0]
z1 = z[1]
z2 = z[2]
z3 = z[3]
z4 = z[4]
z5 = z[5]

# evaluation of the fifth order model
actual_solar_radiation = output_array
estimated_solar_radiation_fifth = z0*pow(input_array, 5) + z1*pow(input_array, 4) + z2*pow(input_array, 3) + z3*pow(input_array, 2) + z4*pow(input_array, 1) + z5
RMSE_fifth = np.sqrt(np.mean((estimated_solar_radiation_fifth-actual_solar_radiation)**2))
print('Root mean square for fifth model = ', RMSE_fifth, 'Watt/m2')


plt.figure(7, figsize=(6, 5))
plt.scatter(ip_vect, op_vect, marker='o', linewidths=0.005, alpha=0.01)
plt.xlabel(r'Image luminance $\mathcal{L}$', fontsize=14)
plt.ylabel(r'Solar irradiance $\mathcal{S}$ [W/m$^2$]', fontsize=14)
line_X = np.arange(3000, 70000)
line_y = z0*pow(line_X, 5) + z1*pow(line_X, 4) + z2*pow(line_X, 3) + z3*pow(line_X, 2) + z4*pow(line_X, 1) + z5
plt.plot(line_X, line_y, '-r', label='Quintic model')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('./results/quintic-model.png', format='png')
plt.show()




# =============================================