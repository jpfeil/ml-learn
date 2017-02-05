#!/usr/bin/env python2.7

from sklearn import cross_validation
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

"""
   #  Attribute                     Domain
   -- -----------------------------------------
   1. Sample code number            id number
   2. Clump Thickness               1 - 10
   3. Uniformity of Cell Size       1 - 10
   4. Uniformity of Cell Shape      1 - 10
   5. Marginal Adhesion             1 - 10
   6. Single Epithelial Cell Size   1 - 10
   7. Bare Nuclei                   1 - 10
   8. Bland Chromatin               1 - 10
   9. Normal Nucleoli               1 - 10
  10. Mitoses                       1 - 10
  11. Class:                        (2 for benign, 4 for malignant)
"""

db = pd.read_csv('breast-cancer-wisconsin.data', header=None)

x = db.iloc[:, 1:10]
y = db.iloc[:, 10]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)


# First I want to look at the different attributes by class

# Clump Thickness
print 'Clump thickness in benign: {}'.format(np.average(db.iloc[:,1][db[10] == 2]))
print 'Clump thickness in malignant: {}'.format(np.average(db.iloc[:,1][db[10] == 4]))

# Plot cell size and cell shape
# The malignant cells seem to be more uniform then the benign samples
plt.scatter(db.iloc[:, 2][db[10]==2], db.iloc[:, 3][db[10]==2], label='benign', color='blue', marker='o')
plt.scatter(db.iloc[:, 2][db[10]==4], db.iloc[:, 3][db[10]==4], label='malignant', color='red', marker='x')
plt.xlabel('Univormity of Cell Size')
plt.ylabel('Univormity of Cell Shape')
plt.legend(loc='upper left')
plt.show()

