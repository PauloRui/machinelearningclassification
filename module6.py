import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import knnplots
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


#Code common to all modeles from module 3 onwards
##NB. The X and yTransformed variables come from the preprocessing in the previous module.
fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray =  numpy.array(dataList)
X = dataArray[:,2:32].astype(float)
y = dataArray[:, 1]

#Creating charts
yFreq = scipy.stats.itemfreq(y)
print yFreq

plt.bar(left = 0, height = int(yFreq[0][1]), color='b')
plt.bar(left = 1, height = int(yFreq[1][1]), color='r')
plt.xlabel("diagnosis")
plt.ylabel("frequency")
plt.legend(['B', 'M'])
plt.show()

#encodes the labels B and M into numerical variables
le = preprocessing.LabelEncoder() # creates the function
le.fit(y) # isnpects the data to find out how many numbers to use
yTransformed = le.transform(y) # transforms the dataset into numerical values

#scatter plot for correlation matrix
correlationMatrix = numpy.corrcoef(X, rowvar=0)
plt.pcolor(correlationMatrix, cmap = plt.cm.coolwarm_r)# the _r reverses the ordering of the color
plt.clim(-1,1)#sets the range of values
plt.colorbar()
plt.show()

#create a scatter plot
plt.scatter(x = X[:, 0], y = X[:, 1], c = y)
plt.xlabel("radius")
plt.ylabel("texture")
plt.show()

#create a scatter plot panel
def scatter_plot(X, y):
    plt.figure(figsize= (2*X,shape[1],2*X,shape[1]))
    for i in range(X,shape[1]):
        for j in range(X,shape[1]):

            plt.subplot(X.shape[1], X.shape[1], i+1+j*X.shape[1])

            plt.subplot()




XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

knnK3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
knnK15 = neighbors.KNeighborsClassifier(n_neighbors = 15)
nbmodel = GaussianNB()

