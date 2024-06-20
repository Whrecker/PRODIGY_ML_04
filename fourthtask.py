from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.ensemble import RandomForestClassifier
directories=[]
for directory in os.listdir("leapGestRecog"):
    for j in os.listdir("leapGestRecog/"+directory):
        directories.append("leapGestRecog/"+directory+"/"+j)
features=[]
labels=[]
for i in directories:
    for image in os.listdir(i):
        if image[9:11]=="01":
            label=1
        elif image[9:11]=="02":
            label=2
        elif image[9:11]=="03":
            label=3
        elif image[9:11]=="04":
            label=4
        elif image[9:11]=="05":
            label=5
        elif image[9:11]=="06":
            label=6
        elif image[9:11]=="07":
            label=7
        elif image[9:11]=="08":
            label=8
        elif image[9:11]=="09":
            label=9
        elif image[9:11]=="10":
            label=10
        image = cv2.imread(os.path.join(i, image))
        if image is not None:
            image_resized = cv2.resize(image,(32,32))
            image_normalized = image_resized / 255.0
            features.append(image_normalized)
            labels.append(label)
X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)
X_train=np.asarray(X_train)
X_test=np.asarray(X_test)
X_train_flattened = X_train.reshape(len(X_train), -1)
X_test_flattened = X_test.reshape(len(X_test), -1)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_flattened, y_train)

predictions = model.predict(X_test_flattened)

accuracy = model.score(X_test_flattened, y_test)
print("Accuracy:", accuracy)

plt.figure(figsize=(10, 8))
num_images = 10
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    if X_test[i].shape[-1] == 3:
        plt.imshow(X_test[i])
    else:
        plt.imshow(X_test[i], cmap='gray')
    plt.title(f"Actual: {y_test[i]}, Predicted: {predictions[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
