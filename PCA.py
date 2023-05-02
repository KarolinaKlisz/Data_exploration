#!/usr/bin/env python
# coding: utf-8

# # PCA
# 

# In[2]:


import pandas as pd
import numpy as np
df = pd.read_csv('./nyt-frame.csv', header = 0)
data = df.iloc[:,9:]
array = data.values
df


# In[3]:


header = list(df.columns.values[9:])
sample_word = np.random.choice(header, 20, replace=False)
print(sample_word)


# In[4]:


from sklearn.decomposition import PCA
pca = PCA()
X_pca = pca.fit_transform(array)


# In[5]:


X_pca


# In[6]:


pc1 = pca.components_[0]
attributes = list(df.columns.values[9:])
sorted_pc1 = sorted(zip(pc1, attributes))
print('--------PC1--------maximum values-----') 
for value, attr in sorted_pc1[-15:]: 
    print(attr, value)

print('--------PC1--------minimum values-----')
for value, attr in sorted_pc1[:15]:
    print(attr, value)


# We would like to see 15 maximum and minimum values. As we can see most maximum values match with art-related terms and most minimum values match with music-related terms. 

# In[7]:


pc2 = pca.components_[1]
attributes = list(df.columns.values[9:])
sorted_pc2 = sorted(zip(pc2, attributes))
print('--------PC2--------maximum values-----') 
for value, attr in sorted_pc2[-15:]:
    print(attr, value)
print('--------PC2--------minimum values-----')
for value, attr in sorted_pc2[:15]:
    print(attr, value)


# As in the previous example we can see that most maximum values match with music-related terms and most minimum values match with art-related terms. 

# Summary: <br>
# - on the x-axis, which corresponds to the first components, music-related terms can be found on the left side while art-related terms can be found on the right side
# - on the y-axis, which corresponds to the second components, art-related terms are located at the bottom, while music-related terms are located at the top

# In[8]:


import matplotlib.pyplot as plt
reds = df['class.labels'] == 'art'
blues = df['class.labels'] == 'music'
plt.figure()
plt.scatter(X_pca[np.array(reds), 0], X_pca[np.array(reds), 1], c="red", label = 'art')
plt.scatter(X_pca[np.array(blues), 0], X_pca[np.array(blues), 1], c="blue", label = 'music')
plt.title("Projection by PCA")
plt.xlabel("1st component")
plt.ylabel("2nd component")
plt.legend()
plt.show()


# In[9]:


variance_ratio = pca.explained_variance_ratio_
plt.plot(variance_ratio, 'ro')
plt.show()
print(sum(variance_ratio[0:10])) 


# In[10]:


dataframe = pd.read_csv('./04cars-data.csv', header = 0)
data1 = dataframe.iloc[:,-11:] #wejście dla PCA
dataframe


# In[11]:


data1


# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data1)
df_scaled = pd.DataFrame(scaled_data, columns=data1.columns) #znormalizowane dane
print(df_scaled)


# In[13]:


pca = PCA()
pca_cars = pca.fit_transform(df_scaled)
pca_cars


# In[15]:


#Absolute value
variance_ratio1 = pca.explained_variance_ratio_
plt.plot(variance_ratio1, 'ro')
plt.title("Absolute value")
plt.show()
print(sum(variance_ratio1[0:2]))
print(sum(variance_ratio1[0:3]))


# In[16]:


attributes = list(df_scaled.columns.values[-11:])
pc1 = pca.components_[0]
pc2 = pca.components_[1]
print('Attribute, PC1, PC2')
for i in range(0, pc1.shape[0]):
    print(attributes[i] + ':' + repr(pc1[i]) + ':' + repr(pc2[i]))


# In[17]:


sorted_pc1 = sorted(zip(pc1, attributes))
print('--------PC1--------') 
for value, attr in sorted_pc1: 
    print(attr, value)

sorted_pc2 = sorted(zip(pc2, attributes))
print('--------PC2--------') 
for value, attr in sorted_pc2: 
    print(attr, value)


# In[18]:


plt.figure(figsize=(20,20))
plt.scatter(pca_cars[:,0], pca_cars[:,1])
for i, model in enumerate(dataframe['Vehicle Name']):
    plt.annotate(model, (pca_cars[i,0], pca_cars[i,1]))
plt.show()


# In[19]:


#visualization
def myplot(score, coeff, labels=None):
    plt.figure(figsize=(25,18))
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs*scalex, ys*scaley)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i,1], color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1]*1.15, "Var" + str(i+1), color = "g", ha = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1]*1.15, labels[i], color = 'g', ha = 'center')

myplot(X_pca[:, 0:2], np.transpose(pca.components_[0:2, :]), attributes)
axes = plt.gca()
axes.set_ylim([-1.0, 1.0])
plt.show()


# In[20]:


from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

lfw_people = fetch_lfw_people(min_faces_per_person=50)

X_train, X_test, y_train, y_test = train_test_split(lfw_people.data, lfw_people.target, test_size=0.3, random_state=42)

X_train


# In[22]:


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

mlp = MLPClassifier()
svm = SVC()

for n_components in [100, 50]:

    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    
    # MLP
    mlp.fit(X_train_pca, y_train)
    y_pred = mlp.predict(pca.transform(X_test))
    print(f"MLP with {n_components} components:")
    mlp_report = classification_report(y_test, y_pred, output_dict=True)
    mlp_df = pd.DataFrame(mlp_report).transpose()
    mlp_df = mlp_df[['precision', 'recall', 'f1-score']]
    print(mlp_df)
    
    # SVM
    svm.fit(X_train_pca, y_train)
    y_pred = svm.predict(pca.transform(X_test))
    print(f"SVM with {n_components} components:")
    svm_report = classification_report(y_test, y_pred, output_dict=True)
    svm_df = pd.DataFrame(svm_report).transpose()
    svm_df = mlp_df[['precision', 'recall', 'f1-score']]
    print(svm_df)

# visualization scree plot
pca = PCA(n_components=100)
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('sum of explained variance')
plt.show()


# We can see that both accuracy and the mean as well as weighted mean are lower for dimensionality reduction to 50,
# and higher for reduction to 100 dimensions.
# We want to choose the optimal number of components that retains enough information but reduces their dimensionality.
# Such a number can be determined based on the point at which the explained variance increment by each subsequent
# component is small. From the plot, we can read that this optimal number is around 50-60 components.
# 

# In[55]:


import matplotlib.pyplot as plt

# PCA train
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)

#Displaying the first 20 principal components in grayscale.
fig, axes = plt.subplots(4, 5, figsize=(15, 12),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='gray')
    ax.set_title(f"Eigenface {i+1}")

plt.show()


# After dimensionality reduction, the original test image can be represented as a combination of weighted eigenfaces
# that form a basis for the face space.
# Eigenfaces allow for finding the most characteristic features of a face, such as the shape of the nose, eyes, mouth,
# and reducing the dimensionality based on these features. This facilitates the classification of images.
# 
