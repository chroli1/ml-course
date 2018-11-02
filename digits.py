from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

digits = load_digits() #Load the digits dataset
X = digits.data #Features matrix
y = digits.target #Target vector

fig, axes = plt.subplots(10, 10, figsize=(8, 8),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

#Visualise the first 100 images in digits:
for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary', interpolation='nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform=ax.transAxes, color='green')

iso = Isomap(n_components=2) #Reduce data to 2d
iso.fit(digits.data)
data_projected = iso.transform(digits.data)

#Visualise the reduced dataset:
plt.figure() #Create new figure
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=digits.target,
            edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5);

#Use Naive Gaussian Bayes to train and predict labels
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
result = accuracy_score(ytest, y_model)
print ('Accuracy score: {}'.format(result))

#Calculate confusion matrix for the above fit:
cmat = confusion_matrix(ytest, y_model)
sns.heatmap(cmat, square=True, annot=True, cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')