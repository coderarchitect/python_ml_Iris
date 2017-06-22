import numpy as np
from sklearn import datasets, linear_model, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

iris = datasets.load_iris()
irisX = iris.data[:, :2]
irisY = iris.target
x_min, x_max = irisX[:, 0].min() - .5, irisX[:, 0].max() + .5
y_min, y_max = irisX[:, 1].min() - .5, irisX[:, 1].max() + .5
step = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step));

# Split into training and testing sets
# Use KNN to predict
np.random.seed(0)
indices = np.random.permutation(len(irisX))
irisXTrain = irisX[indices[:-10]]
irisYTrain = irisY[indices[:-10]]
irisXTest  = irisX[indices[-10:]]
irisYTest  = irisY[indices[-10:]]

#List colors for plotting
colorList  = ["red", "green", "blue"]

# Legend for plot
redPatch   = mpatches.Patch(color='red', label='iris setosa')
bluePatch  = mpatches.Patch(color='green', label='iris versicolor')
greenPatch = mpatches.Patch(color='blue', label='iris virginica')
legend     = [redPatch, bluePatch, greenPatch]

#Get color to use for plot based on flower type
def type_to_color (ftype):
    return colorList[ftype]


def predict_and_plot (model):
    modelType = type(model)
    print ("Model Type", modelType)
    supportedModelTypes = [GaussianNB, KNeighborsClassifier, linear_model.LogisticRegression, svm.SVC]
    if modelType in supportedModelTypes:

        redPatch = mpatches.Patch(color='red', label='iris setosa')
        bluePatch = mpatches.Patch(color='green', label='iris versicolor')
        greenPatch = mpatches.Patch(color='blue', label='iris virginica')
        legend = [redPatch, bluePatch, greenPatch]

        supportedTypeIndex = supportedModelTypes.index(modelType)
        accuracyMsg = titles[supportedTypeIndex] + " Accuracy: "
        model.fit(irisX, irisY)
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]);
        Z = Z.reshape(xx.shape)

        figure = plt.figure(supportedTypeIndex + 1, figsize=(6, 4))
        plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
        plt.scatter(irisXTrain[:, 0], irisXTrain[:, 1], c=colors)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.title("Iris Data Set: "  + titles[supportedTypeIndex])
        plt.grid (True)
        plt.legend(handles = legend)
        yPred = model.predict(irisXTest)
        print(accuracyMsg, accuracy_score(irisYTest, yPred))
        #plotId = plotId + 1
    else:
        print ("Model type is not supported")

#initialize color list
colors = list(map (lambda x: type_to_color(x), irisYTrain))
print (colors)

#Create prediction models
nbModel = GaussianNB (priors=None)
knn     = KNeighborsClassifier(n_neighbors=15);
lrModel = linear_model.LogisticRegression(C=10000)
svmModel = svm.SVC(kernel='poly', degree=3)

titles = ["Gaussian", "KNN", "Logistic Regression", "SVM"]

#Fit and plot models
fittedModels = [predict_and_plot(model) for model in [nbModel, knn, lrModel, svmModel]]

plt.show()

