# CSE 151A Project

# Predicting Diseases

# Introduction 

This project aims to investigate and identify key factors that contribute to the diagnosis of specific diseases. Predicting the presence of various diseases based on a single factor is challenging; therefore, we have endeavored to predict them using multiple features available through machine learning techniques. This project will be carried out using the SUPPORT2 dataset from the UC Irvine Machine Learning Repository. This dataset comprises records of 9,105 critically ill patients from five United States medical centers, collected between 1989-1991 and 1992-1994, as illustrated by [Figure 1](#figure-1-support-project-phases). Each row contains the records of hospitalized patients who met the inclusion and exclusion criteria for one of eight disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. For disease prediction, we selected 21 features from this dataset, such as physiology scores, chemical levels, and various measurements. [Table 1](#table-1-uci-dataset-features-detailed-information) presents the names and descriptions of these features. Machine Learning algorithms—logistic regression, neural networks, decision trees with XGBoost and Gradient Boosting, Support Vector Machines, and K-Nearest Neighbors—are employed, and we will later examine how our results fare in predicting these diseases based on the available data.


## Figure 1: SUPPORT Project Phases

![Figure 1](images/figure1.png)

*Figure 1 — Overall schematic presentation of phases I and II of the Study to Understand Prognoses and Preferences for Outcomes and Risks of Treatment (SUPPORT) project, 1989 to 1994. DNR indicates do not resuscitate; CPR, cardiopulmonary resuscitation; and ICU, intensive care unit.*


## Table 1: UCI Dataset Features Detailed Information

![Table 1](images/table1.png)

*Table 1 - UCI Dataset selected features descriptions.*


The rest of the paper is organized as follows: the Method Section will present the models we executed in the order we explored them. The Results Section will include findings from the aforementioned methods. The Discussion Section will present our interpretations, thought processes, and critiques from beginning to end. Finally, the Conclusion Section will offer our final thoughts.


# Methods: 
#### Data Exploration:

For data exploration, we called the [pd.DataFrame()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=RA1zgeeIgR2p&line=3&uniqifier=1) function to display the variable name and descriptions, and we found that there are 47 variables in total, which is a large dataset.
  Then we created a data frame using [read_csv](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=YC0o6IQ1i-ec&line=1&uniqifier=1), after transferring, we selected 21 columns, with 20 features: ```'age'```, ```'sex'```, ```'scoma'```, ```'race'```, ```'sps'```, ```'aps'```, ```'diabetes'```, ```'dementia'```, ```'ca'```, ```'meanbp'```, ```'wblc'```, ```'hrt'```, ```'resp'```, ```'temp'```, ```'pafi'```, ```'alb'```, ```'bili'```, ```'crea'```, ```'sod'```, ```'ph'```, and ```'dzgroup'``` as our target.
  
  Then we wanted to see the counts of each class (unique values in dzgroup) and plot it in order to see if there was any disparity between the classes. We used seaborn to plot our data on a barplot to easily visualize the different in counts between our targets.

```
df = pd.read_csv('https://archive.ics.uci.edu/static/public/880/data.csv')
df = df[['age','sex','death','dzgroup','scoma','race','sps','aps','diabetes','dementia','ca','meanbp','wblc','hrt','resp','temp','pafi','alb','bili','crea','sod','ph']]
```
  The next step we took was to check for null values in the dataset. We used the [df.isnull().sum()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=KtldNNfpP723&line=1&uniqifier=1) function to calculate the null values for each feature we selected.
  
#### Data Preprocessing: 
  
To deal with the NaN values we found in our dataset, we decided to drop them. We dropped them using 
```
df = df.dropna(axis = 0, how = 'any')
```

  For the features containing value type string, we applied the [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to distinguish.

  For binary features like ```'sex'```, we divided them into integers 0 and 1 using the function
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
```
df['sex'].replace('female', 0, inplace=True)
df['sex'].replace('male', 1, inplace=True)
```
  For nonbinary features, we applied the [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function to get them in a binary format. The column 32 was dropped because it is the nan value from race.
  ```
ohe = OneHotEncoder()
list1 = ['dzgroup', 'race', 'ca']
for i in list1:
    myohedzgroup = ohe.fit_transform(df[i].values.reshape(-1,1)).toarray()
    myohedzgroup=pd.DataFrame(myohedzgroup, columns=ohe.categories_[0])
    df=df.drop([i], axis=1)
    df=pd.concat([df,myohedzgroup],axis=1)

df = df.dropna(axis = 0, how = 'any')
df.drop(df.columns[32], axis=1)
```
  After the above data exploration and preprocessing, we were able to apply some visualization tools to help us explore the pattern of data.



### First Model
#### Multi-Class Logistic Regression
  In order to make it easy to observe some patterns, we decided to apply multi-class logistic regression first.
  
  The first model was not the most precise, as there was a relatively clear sign of underfitting due to the cross-validation score being lower at the start than the training score, and only a relative evening out towards the end of the graph. 
  
  We can possibly improve this model by selecting different features for training use to further finetune the results and not have overfitting or underfitting for the model.
  
  In our first model, we set ```'dzgroup'``` as our target and used the rest of the columns as our features. To ensure that our feature data was normalized, we implemented minmax normalization. We then split the data into training and testing sets, with a 70:30 ratio and a random state of 0. We built eight different logistic regression models for single-class regression to predict each target and reported the results using ```accuracy_score```, ```classification_report```, and ```confusion_matrix```. 
  ```
  for i in targets_ohe.columns:
  X_train, X_test, y_train, y_test = train_test_split(features, targets_ohe[i], test_size=0.3, random_state=0)
  logreg = LogisticRegression(max_iter = 1000, solver = 'liblinear')
  logregmodel = logreg.fit(X_train, y_train)
  yhat_train = logreg.predict(X_train)
  yhat_test = logreg.predict(X_test)

  train_accuracy = accuracy_score(y_train, yhat_train)
  test_accuracy = accuracy_score(y_test, yhat_test)
  print(classification_report(y_test, yhat_test))
  conf_matrix = confusion_matrix(y_test, yhat_test)
  ```
  We also generated learning curves for each logistic regression model, calculating mean training and testing scores across different cross-validation folds for each training and testing size. 
  ```
  train_sizes, train_scores, test_scores = learning_curve(logreg, X, y, cv=5, scoring='accuracy', n_jobs=-1)
  ```
  
  In addition, we built a logistic regression model that predicts multiclass(```'dzgroup'```) and reported the results using ```accuracy_score```, ```classification_report```, and ```confusion_matrix```. We also generated learning curves for the logistic regression model, calculating mean training and testing scores across different cross-validation folds for each training and testing size. Finally, We plotted learning curves for the logistic regression model.
  ```
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=0)
logreg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
logregmodel = logreg.fit(X_train, y_train)
yhat_train = logreg.predict(X_train)
yhat_test = logreg.predict(X_test)
```
  
  Overall, our analysis was thorough and rigorous, ensuring that our results were accurate and reliable about the state of the model.

### Second Model
#### Neural Network
  We used a ```MinMaxScaler``` to apply minmax normalization to our feature data, and for the Keras Sequential Model, we split the data into training and testing sets with an 80:20 ratio, setting the random state to 0 using ```train_test_split```:
  whereas for the hyperparameter tuning model, we split the data into training and testing sets with an 85:15 ratio, setting the random state to 1, in order to train more data to fit our data.
```
# scaling data
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
```
  
  We built the base model (Keras Sequential Model) to predict each target and report the result using ```classification_report```. 
  ```
def buildmodel():
    model = Sequential([
        Dense(units = 72, activation = 'tanh', input_dim = 24),
        Dense(units = 42, activation = 'tanh'),
        Dense(units = 42, activation = 'tanh'),
        Dense(units = 42, activation = 'tanh'),
        Dense(units = 8, activation = 'softmax')
    ])
    model.compile(optimizer ='SGD', loss='categorical_crossentropy')
    return(model)
```

  
When we compiled the model, we used Stochastic Gradient Descent to minimize the error rate and used Categorical Crossentropy as the loss function. When we fit the model, we set the number of epochs to 100, the batch size to 20, the validation split to 0.1, and the verbose to 0.

  We performed K-Fold Cross-Validation with different random splits of the data. The entire cross-validation was repeated 5 times and in each repetition, the data was split into 10 subsets. We also calculated the accuracy for each subset of the data when processing cross-validation. 
  ```
estimator = KerasClassifier(model=buildmodel, epochs=50, batch_size=20, verbose=0) 
kfold = RepeatedKFold(n_splits = 10, n_repeats = 5)
results = cross_val_score(estimator, X_train, y_train, cv=kfold, n_jobs = 1,scoring = 'accuracy')
```

  Then, we built another model(hyperparameter tuning model) to predict each target and report the result using classification report. We set the range of units in the input layer to 18180, the range of units in hidden layers to 12180, and the units in the output layer to 8. The available choices for the activation function include ```'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu', and 'mish'```. The available choices for optimizer include ```'sgd', 'adam', and 'rmsprop.'``` The range of learning rate is ```0.001~0.1```. The available choices for the loss function include 'categorical_crossentropy', 'mse', and 'binary_crossentropy.' After that, we run the model to find the most optimized parameters and use them to rebuild our model.
```
def build_model(hp):
    model = keras.Sequential()
    activation = hp.Choice("activation", ['relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu', 'mish'])
    # input layer
    model.add(
        layers.Dense(units = hp.Int("units", min_value = 18, max_value = 180, step = 20),
              activation = activation,
              input_dim = X.shape[1]
        )
    )
    # hidden layers
    for i in range(3):
      model.add(
          layers.Dense(
              units = hp.Int("units", min_value = 12, max_value = 180, step = 20),
              activation = activation,
          )
      )
    # output layer
    model.add(
          layers.Dense(
              units = 8,
              activation = 'softmax'
          )
      )
    loss = hp.Choice("loss", values = ["categorical_crossentropy", "mse", "binary_crossentropy"])
    learning_rate = hp.Float("lr", min_value = 0.001, max_value = 0.1, step = 0.01)
    optimizer = hp.Choice("optimizer", values = ["sgd", "adam", "rmsprop"])
    if optimizer == "sgd":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "adam":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer = optimizer,
        loss = loss,
        metrics = ["accuracy"],
    )
    return model

```

```
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=10,
    overwrite=True,
    directory="my_dir",
    project_name="hypertune",
)
tuner.search(X_train, y_train, epochs=50, validation_split = 0.2, callbacks = [early_stopping], verbose = 0)
```
  The last thing we do is we apply OverSamplying (SMOTE) to our train datasets. Then, applying the oversampling data to our best model from our hyperparameter tuning. Then, report the result using a classification report and plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting.

```
smote = SMOTE(random_state = 21)
X_train_resample, y_train_resample = smote.fit_resample(X_train, y_train)
```

### Third Model
#### SVM

  For the third model, we tried to implement the SVM model, since SVM supports both numerical and categorical data, so it was more flexible in data processing compared to Naive Bayes. First, we did the data processing for our target using one-hot encoding as we did on the initial data processing part. We also encoded 'race' and 'sex' by one-hot encoding and manually separating the values. Then we scaled our data, using both ```MinMaxScaler()``` and ```StandardScaler()```. 
  
  The first time we tried ```MinMaxScaler()``` after implementing the SVM model, we found the accuracy was low, so we then used ```StandardScaler()```. However, ```StandardScaler()``` needs to reprocess the data, since we had one-hot encoded features, and ```StandardScaler()``` is only used for numerical data. To separate the numerical data from the categorical data, we utilized ```iloc[:, 19:]``` to extract the categorical encoded features, and dropped the ```'sex'``` feature. After getting the scaled X, we combined them using np.concatenate() methodto get the complete X_scaled. Before we passed the data to the SVM model, we split the data with a ratio of train:test = 80:20 using ```train_test_split```. 
  
  Then we tried our first SVM model, with set parameters ```C = 0.1``` , ```degree = 2```, and ```kernel ='poly'```. When we fit our SVM model, we needed to transform ```y_train``` to a 1-D array, so we passed ```y_train.idxmax(axis=1).values``` as the parameters. We also used ```.idxmax(axis=1).values``` to get ```t_true``` in order to pass the data to ```classification_report```. However, the result of our SVM model was a low accuracy score, so to improve the accuracy, we decided to try different parameters to build our SVM model. The second time, we used ```GridSearch``` to find optimized parameters. We set our grid as: ```param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}```.

1. Use of the ```'poly'``` kernel:

```
svm = SVC(kernel ='poly', degree = 2)
svm.fit(X_train,y_train)

y_true = y_test
y_pred = svm.predict(X_test)

print(classification_report(y_true, y_pred, zero_division=0))
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")
```

2. Hyperparameter tuning using ```GridSearch```: 

```
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}

svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train,y_train.idxmax(axis=1).values)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

accuracy = best_model.score(X_test, y_test.idxmax(axis=1).values)
print(f"Best Hyperparameters: {best_params}")
print(f"Accuracy on Test Set: {accuracy}")
```

We also used oversampling with ```SMOTE``` and ```RandomOverSampler```, but it also resulted in a low accuracy, as shown in the results section.

1. Use of ```SMOTE```:

```
smote = SMOTE(random_state=21)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train.idxmax(axis=1).values)
unique_classes_resampled, class_counts_resampled = np.unique(y_train_resampled, return_counts=True)

for class_label, count in zip(unique_classes_resampled, class_counts_resampled):
    print(f"Frequency of Class {class_label}: {count} instances")
```

2. Use of ```RandomOverSampler```: 

```
rs = RandomOverSampler(random_state=11)
X_train_resampled, y_train_resampled = rs.fit_resample(X_train, y_train.idxmax(axis=1).values)
unique_classes_resampled, class_counts_resampled = np.unique(y_train_resampled, return_counts=True)

for class_label, count in zip(unique_classes_resampled, class_counts_resampled):
    print(f"Frequency of Class {class_label}: {count} instances")
```

#### Decision Tree Learning
  Another model we tried for the third model was the ```Decision Tree Learning``` model.

  The data preprocessing was the same as processing for SVM, where we applied encoding and ```StandardScaler``` to make our data clean, but this time we tried ```XgBoost``` model, setting the parameters as follows: 

```
eval_set = [(X_train, y_train), (X_test, y_test)]
model = xgb.XGBClassifier(objective='multi:softmax', max_depth=2, learning_rate=0.1, n_estimators=100, eval_metric='mlogloss')
model.fit(X_train, y_train, eval_set=eval_set, verbose=0)

results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])

train_error = results['validation_0']['mlogloss']
test_error = results['validation_1']['mlogloss']
```


We then decided to try different parameters as well to improve the accuracy by ```RandomizedSearchCV```. As the diagram shows below, we selected four parameters: ```'max_depth'```, ```'learning_rate'```, ```'n_estimators'```, and ```'subsample'```:

```
model = xgb.XGBClassifier()

param_dist = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=25, scoring='accuracy', cv=3, verbose=1, random_state=0)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
print("Best accuracy found: ", random_search.best_score_)
```

We also tried another parameter search method, ```GridSearchCV```, in order to get the best model parameters.
In order to minimize error, we used the same parameters as ```RandomizedSearchCV```. 

```
model = xgb.XGBClassifier()

param_grid = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3, verbose=1)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best accuracy found: ", grid_search.best_score_)
     
```



### Gradient Boosted Tree
Gradient Boosted Tree was the third method we chose.
We manually tried one set of parameters for the model as ```n_estimators=100, learning_rate=0.1, max_depth=3, random_state=21```. After training the model,  we used ```classification_report``` to evaluate the model.

```
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=21)

model.fit(X_train, y_train)

train_error = [log_loss(y_train, y_pred_proba) for y_pred_proba in model.staged_predict_proba(X_train)]
test_error = [log_loss(y_test, y_pred_proba) for y_pred_proba in model.staged_predict_proba(X_test)]
```

### KNN Model
KNN was our final version for the third model. We simply applied the ```KNeighborsClassifier``` function, and below are the details about our parameters in the KNN model:

```
k = 10
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

y_true = y_test
y_pred = knn_classifier.predict(X_test)
print(classification_report(y_true, y_pred, zero_division = 0))
```

# Results : 
### Data Exploration
![Screenshot 2024-03-15 at 9 15 24 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/ac763465-a001-48d8-914a-12a3bed1d63e)

![Screenshot 2024-03-15 at 9 16 27 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/45bf61de-303e-4b75-8055-55ed719e477a)

![Screenshot 2024-03-15 at 9 17 03 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/6c31f9d2-325f-4bb0-b1ab-b0cd883556df)

All 47 of the datasets features

After understanding the meaning of different features, and from the output data we got after printing the entire dataset, it shows we have imbalanced data: we have some null values, and some features that are not clean enough to be used for training models, which indicates that we need to do some data preprocessing:

![Barchart](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/66afc575-09cd-4fc2-b152-805c58aabf06)

Counts of each of our targets

![Screenshot 2024-03-15 at 8 42 12 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/5a15641a-fd11-4260-952f-42b6d960342b)

Plot of features (out of the ones we chose) which contained NaN values as well as how many

![11111](https://github.com/sebastian-dv/CSE-151A-Project/assets/147887997/f776e7ce-7da9-4b16-be76-5e1c89e2a4fa)

Heatmap of the entire dataset

### Data Preposessing

<img width="1316" alt="截屏2024-03-15 下午8 52 29" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/147887997/e07aa042-c77c-4764-a521-cf22d6c816cd">

The final preprocessed dataframe.

### First Model
#### Multi-Class Logistic Regression
![Screenshot 2024-03-10 at 7 16 17 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/90e3f1ea-bb2b-4f66-ad46-c7322d1a7d89)

This is our plot for our multi-class regression model, comparing our training score vs our cross-validation score.

![Screenshot 2024-03-10 at 7 20 21 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/f9e92ef8-55ad-4b95-ad17-ad58568ef99d)

Here is one of the single-target models, and when comparing the multi-target results with the single-target results, the model itself doesn't look too different and the results are relatively the same.

### Second Model
#### Neural Network
  <img width="994" alt="截屏2024-03-14 17 50 05" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c7520ece-e0ed-410f-a305-d7d765f0e68c">
  
  Base model classification report.
  
  <img width="620" alt="截屏2024-03-14 17 51 48" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/a32b5785-6b80-4285-9f65-3a5f06e61ee9">
  
  Base model train/validation error.
  
  <img width="796" alt="截屏2024-03-14 18 0319" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/b924ae2b-fda3-4a52-91ea-178df74ede69">
  
  10-fold cross validation accuracy.
  
  <img width="632" alt="截屏2024-03-14 17 59 17" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/a065caa2-7af9-4242-a78f-21a66eca88bf">
  
  Hyperparameter tuning model classification report and train/validation error.
  
  <img width="491" alt="截屏2024-03-14 17 59 55" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c8496de3-3381-4072-b4bd-31ebfb36d64c">
  
  Oversampling model classification report.
  
<img width="611" alt="截屏2024-03-14 18 00 25" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/4ffb68b8-b8a0-4271-9ce4-23903a450315">

Oversampling model train/validation error.

### Model 3

#### SVM
  Base Model Performance Train:

  <img width="452" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/8b045ba9-758a-4aca-b03f-3fa164fbde0c">

  Base Model Performance Test:
  
  <img width="466" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/49860168-edae-4775-b669-3b3fda45c5c8">

  Evaluation after Gridsearch:
  
  <img width="494" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/9a30e385-0f18-46d2-a314-c667c504f782">

  Evaluation after ```RandomOverSampler```:
  
 <img width="569" alt="截屏2024-03-14 17 33 57" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c5e7741f-5925-427d-a3aa-2639c6649272">

#### XGboost

  Base Model Performance Train:
  
  <img width="461" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/a12d7df8-0627-400c-acde-9316cb3dcfbd">

  Base Model Performance Test:
  
  <img width="455" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/8e69a843-4ad8-43a6-951f-9a6b474bf834">

  Evaluation after Gridsearch:

 <img width="832" alt="截屏2024-03-14 17 11 10" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/69580227-0f0c-415c-b913-431d657bc45a">

 Log-loss fitting curve:
 
 <img width="902" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/68130529/364e37ca-ea56-4faa-bf2c-3ce3e34c0cb0">

### Gradient boosted Tree

Base Model Classification Report

<img width="456" alt="截屏2024-03-14 17 21 33" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c6087504-a4cf-4818-9333-3fc70bab96c1">

Result after oversampling data:

<img width="439" alt="截屏2024-03-14 17 28 22" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/75220e47-7dab-40ce-9f42-89c85be67dbc">

### KNN Model

KNN model classification report

<img width="504" alt="截屏2024-03-14 17 39 16" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/5ba49df3-f499-47c9-af02-abac6b840a47">


# Discussion: 
## How we chose the model:

### Data Exploration

Our objective was to find a populous dataset with multiple indicators of health as well as firm confirmations of the existence of diseases based on those indicators. Therefore, the SUPPORT2 dataset looked exactly like what we were searching for. It had all of the indicators, like ```’age’```, ```’sex’```, ```’race’```, ```’heart rate’```, ```’respiration rate’```, ```’temperature’```, and more that we could use for our targets during model construction and training. However, through further inspection at some of the classes, we found that some had missing values, with some columns having far more null values than complete values. 

Similarly, through our EDA, we found that the number of target classes within ```‘dzgroup’``` were imbalanced. Some of the target diseases appeared much more than others, which we knew could lead to some issues in the future and as a result we might have to perform some oversampling in order to allow our models to better learn from our data.

Prior to preprocessing, we found the total number of NaN values present in our chosen features and plotted them in a bar plot. The ones with seemingly no NaN values had very few which is why the bar cannot be seen for those. We had to figure out how to deal with these NaN values and with so many, it was difficult to decide what exactly to do.


### Data Preprocessing

  For the preprocessing step, our goal was to properly partition the dataset into two groups: useful data and extraneous information. We decided to fixate on the following columns, finding that they were more relevant for our model and project goal: ```'age'```, ```'sex'```, ```'dzgroup'```, ```'scoma'```, ```'race'```, ```'sps'```, ```'aps'```, ```'diabetes'```, ```'dementia'```, ```'ca'```, ```'meanbp'```, ```'wblc'```, ```'hrt'```, ```'resp'```, ```'temp'```, ```'pafi'```, ```'alb'```, ```'bili'```, ```'crea'```, ```'sod'```, and ```'ph'```. We then tried to isolate the target values that we thought would be the best for results, and so we focused on the column ```‘dzgroup’```, since it contained important information like rate of colon cancer, coma, lung cancer, and more, all of which fell under the targets we were looking for. Finally, after securing our features and targets, we looked to make the entire dataset all readable data, so we dropped every row of data containing a null value to ensure the data was properly aligned and evenly spread across all features and targets. At the point of completing preprocessing, we were satisfied with the resulting dataset we got, as there were still plenty of entries to properly train each model. However, looking back now, maybe it would have been better to keep some of the null values, since it would have been better at training models even at the expense of exposing it to null values that could mess up the training.

### After Data Preprocessing:

  We chose to use a ```multi-class logistic regression model``` to train and fit the model with the preprocessed data. We then used the results to test for overfitting vs underfitting, accuracy, and error for train vs test data. We are thinking of testing with a ```multi-class classification model``` and a ```Keras Sequential model``` next to look for better results. This is because we need models that are capable of outputting multiple different classes since our targets are multiple different diseases. These next two models should be more powerful and hopefully better at predicting our targets.
  
  In conclusion, we thoroughly analyzed the description of each variable to understand their significance, carefully selecting features pertinent to our topic to ensure effective data preprocessing. We meticulously printed out the data frame containing the chosen features, meticulously verifying its shape to ascertain the number of observations accurately. We conducted a meticulous check for any empty values, such as null, within the data frame, ensuring data integrity before proceeding. Additionally, for each categorical attribute, we meticulously listed out all unique elements, ensuring comprehensive understanding and meticulous preprocessing of the data.

### Model 1

  For our Model1, our goal is to build a baseline model that can help us understand the dataset better, and serve as comparison for our more complex future models. We decided to implement a logistic classifier for its ease of implementation and high interpretability. By default, logistic classifiers are designed for binary classification. However, using the parameter multi_class='multinomial', they are able to handle multi-class classification.

  Our Model1 achieved 0.55 training accuracy and 0.54 testing accuracy, with the cross-validation score also being very similar to the training score. Based on these observations, we concluded that no overfitting occurred. 

  Although the accuracy is not great, at this point, we were satisfied with Model 1 and its performance. We believed that the low accuracy was mainly due to the poor choice of model as a logistic classifier is a binary classification algorithm at the end of the day. With more advanced models and careful tuning, we should be able to level up that accuracy in the future.

### Model 2

  For Model 2, we decided to go with something much more powerful, a Neural Network. 

  We started out by building a base model that has 4 hidden layers and 1 output layer. From our experiences in HW2, we know that softmax for output activation and categorical cross entropy for loss are very solid choices in a multi-class setting. We then experimented with a couple of different activation functions and chose tanh which yielded relatively good performance. This base model only achieved 0.55 testing accuracy, which is definitely lower than we expected. We also saw signs of overfitting, since our training loss is much lower than our validation loss, leading us to believe that our real accuracy is likely to be even lower. 

  To get a sense of how our model truly performs to unseen data, we utilize K-fold cross-validation. Our K-fold cross-validation score was indeed lower at 0.52 accuracy. 

  In an attempt to improve the accuracy, we used hyperparameter tuning. Since we’re certain that softmax is the correct output activation function, we tuned the activation function for the rest of the layers, number of units, learning rate, and loss functions. At first, we attempted a large range of values on choosing the number of units in each layer, and the train/validation loss graph shows the result was extremely overfitting. We also increased the number of max trials to check if it could help the tuner explore more combinations, but we found out that such an attempt would increase the validation error as well. We proposed that lowering the number of units and layers might reduce the sign of overfitting, so we attempted multiple combinations to find a fitting graph. After tuning, the accuracy remained low at 0.52, with no overfitting this time according to our graph.

  As HP tuning did not improve our accuracy, we resorted to another technique we learned in class, which is oversampling. The reason we chose this technique is because we have an extremely imbalanced dataset, our most populated target has 1725 entries while the least populated target has 98 entries. However, Oversampling actually decreased our accuracy, and our accuracy decreased to 0.38 after oversampling. We believe there are two main reasons why oversampling did not work. One is that our dataset is too imbalanced and the other is that we have a small dataset. Our model is overfitting after oversampling due to these reasons.

  Model 2 overall was not ideal, we expected much more out of our Neural Network model. We suspect that advanced feature engineering may be needed and potentially changing the number of layers in our sequential model can also be helpful.

### Model 3

  We first tried the SVM model since the model is good for One-Hot encoding targets compared to the Naive Bayes model. Also, according to the resource: Kernel SVMs can effectively handle non-linear decision boundaries, making them useful for tasks where data is not linearly separable. [What are the advantages and disadvantages of using a kernel SVM algorithm? - Quora](https://www.quora.com/What-are-the-advantages-and-disadvantages-of-using-a-kernel-SVM-algorithm), So we decided to try the SVM first.
After we got the result from the classification report of the SVM tuner and OverSampling, although the accuracy score is fair, we found that SVM is limited by choosing an appropriate kernel or manually transforming features to capture non-linear relationships effectively. So we think we may need to try other models to do the comparison. 

  One of our members visited office hours and the Professor suggested we try XGboost. XGboost offers some very attractive features for us. It incorporates regularization, handles missing values, and is able to handle unbalanced datasets. On top of that, it is fast and achieves high accuracy. However, after training our Xgboost classifier, there was not a significant improvement over other models with its 0.57 testing accuracy. We tried random search and grid search to obtain the best parameters, but neither were that effective, improving the accuracy to 0.58.

  In the meantime, we also tried Gradient Boosted Tree as an alternative to XGboost, but the results were not as good. We also displayed the ranking of importance of each feature, results are shown below:
  <img width="717" alt="截屏2024-03-15 18 27 35" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/2bc722a5-6367-4467-939a-36a2c2b67c9b">

  As the picture shows, surprisingly, the ```race``` is the most important feature to discuss about towards our target.


  At this point, we have tried essentially every model besides KNN that has been discussed in this class. We ended up trying KNN as well. Our KNN model only yielded 0.5 testing accuracy, quite a bit worse than XGboost.



# Conclusion

  We believe there are several reasons why our models did not perform that well. 
  
1. imbalanced dataset. Some of our classes have significantly fewer instances than others. For example, Colon Cancer has 98 instances while ARF/MOSF w/Sepsis has 1725 instances. Classifier can be biased toward majority groups
 
2. Insufficient data. After preprocessing and dropping null values. Our dataset only has 3840 entries, which is likely not enough for such a complex classification problem.

3. Feature engineering. Our group put a lot of emphasis on model selection and model tuning, but our feature engineering was very limited, using only the LabelEncoder, OneHotEncoder, and Scalers.

  However, this isn’t to say that our model didn’t improve throughout the project timeline. Although our first logistic classifier model also achieved 50% accuracy, it did so by ignoring some minority classes. In the classification report, we can see that the precision for Colon Cancer is 0. If we look at our final model, although the accuracy overall didn’t improve significantly, the accuracy is now evenly distributed between classes, there are no classes that are very inaccurate. We believe that is a major improvement. 

If we started our project over, there are several things we wished we could have done:

1. Instead of dropping any row that contains null value, maybe we should try to impute them to obtain more data for our model

2. Allocate more time for feature engineering. Good features can significantly enhance a model's ability to capture patterns and relationships in the data.

3. Play around with Neural Network’s number of layers. Although we spent a lot of time tuning Neural Network parameters, we might have benefited from experimenting with different numbers of hidden layers


# Collaboration section: 

1. Name: Pranav Prabu

   Contribution: Abstract, Preprocessing, Model 1, Final Report
2. Name: Sebastian Diaz

   Contribution: Abstract, Preprocessing, Model 1, Final Report
3. Name: Jou-Chih Chang

   Contribution: Abstract, Model 2, Model 3, Final Report
4. Name: Juan Yin

   Contribution: Abstract, Model 1, Model 3, Final Report
5. Name: Irving Zhao

   Contribution: Abstract, Model 2, Model 3, Final Report
6. Name: Xianzhe Guo

   Contribution: Abstract, Model 1, Model 2, Final Report

7. Name: Tiankuo Li
   
   Contribution: Abstract, Model 1, Model 3, Final Report

# Colab Files of Our 3 Models
[Preprocessing](https://colab.research.google.com/drive/1nzW6bMa3XklLFByw_Fc9XWii4gMwAa67?usp=sharing)

[Model 1](https://colab.research.google.com/drive/1PFt7mk4PJi3zmMCn9rKbJEDA61fFsDZk?usp=sharing)

[Model 2](https://colab.research.google.com/drive/1GBM_WtSZDAe_ifttldpFGWvh5ttAP842?usp=sharing)

[Model 3](https://colab.research.google.com/drive/1X1-l40jQnPeq46wu0CgNISCQnN9mFe1C?usp=sharing)
