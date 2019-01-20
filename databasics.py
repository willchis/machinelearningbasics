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

#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) # pandas plotting
#plt.show() # a pyplot function

# histograms
#dataset.hist()
#plt.show()

#scatter_matrix(dataset)
#plt.show()

# Split-out validation dataset
array = dataset.values
mainData = array[:,0:4] # all rows, columns 0-4. All the size data

classData = array[:,4] # all rows, just column 5. Just the class name e.g. "Iris-setosa"

validation_size = 0.20
seed = 7

# model_selection is an sklearn wrapper that helps manip machine learning datasets
# train_test_split is a helper that splits data into random train and test subsets. We're using 0.2 here to get a 80%/20% split
mainData_train, mainData_validation, classData_train, classData_validation = model_selection.train_test_split(mainData, classData, test_size=validation_size, random_state=seed)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

scoring = 'accuracy'
# evaluate each model in turn
results = []
names = []
print('-----------')
print('Model MEAN  (STD)')
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, mainData_train, classData_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111) # this just sets up the plot as 1x1 and is plot no.1.
plt.boxplot(results) # cross validation results from testing models.
ax.set_xticklabels(names)
#plt.show()

# Results show that the KNN model had best accuracy.
# Now make predictions using this model
knn = KNeighborsClassifier()
knn.fit(mainData_train, classData_train)
predictions = knn.predict(mainData_validation) # using the validation dataset to get predictions
print(predictions) # a list of the predictions e.g. I think this will be an 'Iris virginica'.
print(accuracy_score(classData_validation, predictions)) # Check the actual class Data against the predictions
print(confusion_matrix(classData_validation, predictions))
print(classification_report(classData_validation, predictions))

single_prediction = [[4.8, 3.5, 1.2, 0.2]]

print('Based on model, a flower with these dimensions will be an: ' + knn.predict(single_prediction))