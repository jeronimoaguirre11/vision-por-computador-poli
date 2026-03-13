import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Dataset500', 'Dataset500'))

Categories=['Cat','Dog']
flat_data_arr=[] #input array
target_arr=[] #output array

#path which contains all the categories of images
for i in Categories:
    print(f'loading... category: {i}')
    path=os.path.join(datadir, i)
    for img in os.listdir(path):
      img_array=imread(os.path.join(path, img))
      img_resized=resize(img_array, (150, 150, 3))
      if img_resized.flatten().shape == (67500, ):
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(i))
    print(f'loaded category: {i} successfully')

#Convert to numpy
flat_data=np.array(flat_data_arr)
target=np.array(target_arr)

#dataframe
df=pd.DataFrame(flat_data)
df['Target']=target
print(df.shape)

#input data
x=df.iloc[:,:-1]
#output data
y=df.iloc[:,-1]

# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

# Creating a support vector classifier
pipeline = Pipeline([
    ('pca', PCA()),
    ('svm', svm.SVC())
])

param_grid = {
    'pca__n_components': [100, 200, 300],
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf'], # trabajar solo con kernel rbf para mejorar la precisión
    'svm__gamma': ['scale'] # trabajar solo con gamma 'scale' para mejorar la precisión
}

grid = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(x_train, y_train)

y_pred = grid.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy mejorado:", accuracy)
print(classification_report(y_test, y_pred))

# Print the accuracy of the model
print(f"The model is {accuracy * 100}% accurate")
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))

rand_index = np.random.randint(0, len(y_test))
rand_img = x_test.iloc[rand_index].values.reshape(150, 150, 3)
rand_class = y_test.iloc[rand_index]
pred_class = y_pred[rand_index]

plt.imshow(rand_img)
plt.title(f"Predicted: {Categories[pred_class]}, Actual: {Categories[rand_class]}")
plt.show()