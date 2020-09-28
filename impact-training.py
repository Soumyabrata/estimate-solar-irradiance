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
from SG_solarmodel import *





# Collect model files
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

print (['The unique F-numbers of the camera are: ', np.unique(fn_cat)])

# Collect only the proper model files, where F-number is not relatively extremely large
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


# Impact of training- and testing- images on the results
# We propose the cubic model in our paper
one_vector = []
for item in ip_vect:
    one_vector.append(item[0])

two_vector = []
for item in op_vect:
    two_vector.append(item[0])

one_vector = np.array(one_vector)
two_vector = np.array(two_vector)

total_items = len(one_vector)
print (total_items)

no_of_experiments = 100

# Training images evaluation
file_name1 = './cubicbox/' + 'training.txt'
text_file1 = open(file_name1, "w")
# Header line
text_file1.write("serial_number, percentage_of_image, RMSE \n")


# Testing images evaluation
file_name2 = './cubicbox/' + 'testing.txt'
text_file2 = open(file_name2, "w")
# Header line
text_file2.write("serial_number, percentage_of_image, RMSE \n")

for percent_of_image in range(0 + 5, 100 + 5, 5):

    print('Performing experiment for ', str(percent_of_image), '%')
    if (percent_of_image == 100):
        percent_of_image = 99

    no_of_image = round((percent_of_image / 100) * total_items)
    for kot in range(0, no_of_experiments):
        # Choose random training indexes
        training_index = np.random.choice(range(total_items), no_of_image, replace=False)

        ip_regression = one_vector[training_index]
        op_regression = two_vector[training_index]


        # Fit line using training data
        z = np.polyfit(ip_regression, op_regression, 3)
        z0 = z[0]
        z1 = z[1]
        z2 = z[2]
        z3 = z[3]

        # Evaluation
        actual_solar_radiation = op_regression
        estimated_solar_radiation_cube = z0 * pow(ip_regression, 3) + z1 * pow(ip_regression, 2) + z2 * pow(ip_regression,
                                                                                                        1) + z3
        RMSE_training = np.sqrt(np.mean((estimated_solar_radiation_cube - actual_solar_radiation) ** 2))
        text_file1.write("%s,%s,%s\n" % (kot, percent_of_image, RMSE_training))


        # Evaluation of testing images
        test_solar = two_vector
        test_solar = np.delete(test_solar, training_index)

        test_lumi = one_vector
        test_lumi = np.delete(test_lumi, training_index)

        actual_solar_radiation = test_solar
        estimated_solar_radiation_cube = z0 * pow(test_lumi, 3) + z1 * pow(test_lumi, 2) + z2 * pow(test_lumi,1) + z3
        RMSE_testing = np.sqrt(np.mean((estimated_solar_radiation_cube - actual_solar_radiation) ** 2))
        text_file2.write("%s,%s,%s\n" % (kot, percent_of_image, RMSE_testing))


text_file1.close()
text_file2.close()




