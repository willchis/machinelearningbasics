# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print('Dimensions: records = ' + str(dataset.shape[0]) + ' fields = ' + str(dataset.shape[1]))

# head - intro of data
print(dataset.head(20))

print(dataset.describe())

print(dataset.groupby('class').size()) # dataset is a pandas DataFrame object

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) # pandas plotting
plt.show() # a pyplot function