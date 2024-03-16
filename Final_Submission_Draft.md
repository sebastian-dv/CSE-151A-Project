# CSE 151A Project

# Abstract about the course and our project

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
#### Data Preprocessing: 
  For data exploration, we called the [pd.DataFrame()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=RA1zgeeIgR2p&line=3&uniqifier=1) function to display the variable name and descriptions, and we found that there are 47 variables in total, which is a large dataset.
  Then we created a data frame using [read_csv](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=YC0o6IQ1i-ec&line=1&uniqifier=1), after transferring, we selected 21 columns, with 20 features: ```'age'```, ```'sex'```, ```'scoma'```, ```'race'```, ```'sps'```, ```'aps'```, ```'diabetes'```, ```'dementia'```, ```'ca'```, ```'meanbp'```, ```'wblc'```, ```'hrt'```, ```'resp'```, ```'temp'```, ```'pafi'```, ```'alb'```, ```'bili'```, ```'crea'```, ```'sod'```, ```'ph'```, and ```'dzgroup'``` as our target.
  The next step we took was to check for null values in the dataset. We used the [df.isnull().sum()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=KtldNNfpP723&line=1&uniqifier=1) function to calculate the null values for each feature we selected.
  The result of df.isnull().sum() showed that some features contained several null values, and so we decided to deal with this data by removing them.
  For the features containing value type string, we applied the [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to distinguish.
  For binary features like ```'sex'```, we divided them into integers 0 and 1 using the function
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
  For nonbinary features, we applied the [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function to get them in a binary format.
  After the above data exploration and preprocessing, we were able to apply some visualization tools to help us explore the pattern of data.
##### Data Preprocessing: 
1. We printed out a correlation matrix plot of the data frame in the form of a heatmap.
2. We printed out the count of the unique elements in the ```'dzgroup'``` column in the form of a bar plot.
3. We one-hot encoded all of the categorical attributes.
4. After one-hot encoding, we dropped all of the original categorical attributes and all of the empty values.

#### Visualization Tool
1. Parallel Coordinates Plot: we applied this function to visualize the relationship between ```'dementia'``` and other features.

2. Multiple Line Plots: this function was used to check the pattern of two particular features.
   - First, we applied an ```'age'```-```'diabetes'``` pair, which shows people between 40 to 80 are the main group to have diabetes
   - Second, we applied the ```'bili'```-```'hrt'``` pair and ```'bili'```-```'ph'``` pair, their diagram has a similar pattern, and We think we should implement more data to see the pattern between them.

3. We applied the pairplot for the entire dataset twice, before and after we split the data using one-hot encoding.

### Multi-Class Logistic Regression - First Model:
  In order to make it easy to observe some patterns, we decided to apply multi-class logistic regression first.
  
  The first model was not the most precise, as there was a relatively clear sign of underfitting due to the cross-validation score being lower at the start than the training score, and only a relative evening out towards the end of the graph. 
  
  We can possibly improve this model by selecting different features for training use to further finetune the results and not have overfitting or underfitting for the model.
  
  In our first model, we set ```'dzgroup'``` as our target and used the rest of the columns as our features. To ensure that our feature data was normalized, we implemented minmax normalization. We then split the data into training and testing sets, with a 70:30 ratio and a random state of 0. We built eight different logistic regression models for single-class regression to predict each target and reported the results using ```accuracy_score```, ```classification_report```, and ```confusion_matrix```. We also generated learning curves for each logistic regression model, calculating mean training and testing scores across different cross-validation folds for each training and testing size.   
  In addition, we built a logistic regression model that predicts multiclass(```'dzgroup'```) and reported the results using ```accuracy_score```, ```classification_report```, and ```confusion_matrix```. We also generated learning curves for the logistic regression model, calculating mean training and testing scores across different cross-validation folds for each training and testing size. Finally, We plotted learning curves for the logistic regression model.
  
  Overall, our analysis was thorough and rigorous, ensuring that our results were accurate and reliable about the state of the model.

#### First Model Visualization 
![Screenshot 2024-03-10 at 7 16 17 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/90e3f1ea-bb2b-4f66-ad46-c7322d1a7d89)
This is our plot for our multi-class regression model, comparing our training score vs our cross-validation score.
This plot shows that there is some underfitting in our model and thus a logistic regression model likely isn't the best model we could use for our data, which is useful to know for our future models. We also had other similar plots for our logistic regression models that we did for each target rather than multiple at once.

![Screenshot 2024-03-10 at 7 20 21 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/f9e92ef8-55ad-4b95-ad17-ad58568ef99d)
Here is one of single-target models, and when comparing the multi-target results with the single-target results, the model itself doesn't look too different and the results are relatively the same.

### Second Model
#### Neural Network
  We used a ```MinMaxScaler``` to apply minmax normalization to our feature data, and for the Keras Sequential Model, we split the data into training and testing sets with an 80:20 ratio, setting the random state to 0 using ```train_test_split```:
  whereas for the hyperparameter tuning model, we split the data into training and testing sets with an 85:15 ratio, setting the random state to 1, in order to train more data to fit our data.
  
  We built the base model (Keras Sequential Model) to predict each target and report the result using ```classification_report```. The model and results are shown below: 
  <img width="994" alt="截屏2024-03-14 17 50 05" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c7520ece-e0ed-410f-a305-d7d765f0e68c">
  We also compared the loss between training and validation, which shows a stark difference in loss rate, representing the fact that the model was nowhere near the perfect fit: 
  <img width="620" alt="截屏2024-03-14 17 51 48" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/a32b5785-6b80-4285-9f65-3a5f06e61ee9">

  After the basic model, we designed a 5-layer artificial neural network using the ```tanh``` activation function in each layer and the ```softmax``` activation function in the output layer. The number of nodes in the first layer was 72, the number of nodes in the output layer was 8, and the number of nodes in the remaining hidden layers of the mode was 42, as shown below:
  
  <img width="770" alt="截屏2024-03-14 17 52 33" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/d9d6c84a-75d8-4aca-97d9-ad20cec7366c">
  
When we compiled the model, we used Stochastic Gradient Descent to minimize the error rate and used Categorical Crossentropy as the loss function. When we fit the model, we set the number of epochs to 100, the batch size to 20, the validation split to 0.1, and the verbose to 0.
We plotted the linear graph for the training and validation loss of the model we built to see the performance of the model as well as if it was overfitting/underfitting, and specific ```MSE``` and ```accuracy``` as the metrics. When we fit the model, we set the number of epochs to 50, the batch size to 20, and the verbose to 0.

  We performed K-Fold Cross-Validation with different random splits of the data. The entire cross-validation was repeated 5 times and in each repetition, the data was split into 10 subsets. 
  We calculated the accuracy for each subset of the data when processing cross-validation. Then, took the mean of it to see the performance of our model. The model and results are shown below:
  <img width="796" alt="截屏2024-03-14 18 03 19" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/b924ae2b-fda3-4a52-91ea-178df74ede69">

  Then, we built a hyperparameter tuning model to predict each target and report the result using ```classification_report```. We set the range of units in the input layer to 18-180, the range of units in hidden layers to 12-180, and the units in the output layer to 8. The available choices for the activation function were ```'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu', and 'mish'```. The available choices for optimizer were ```'sgd', 'adam', and 'rmsprop.'``` The range of learning rate was ```0.001~0.1```. Finally, the available choices for the loss function were 'categorical_crossentropy', 'mse', and 'binary_crossentropy.' After setting the parameters for testing, we ran the model to find the most optimized parameters and used them to rebuild our model. The hyperparameter tuning model was set up as shown:
  <img width="904" alt="截屏2024-03-14 17 58 18" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/292037a1-333c-47e1-9235-cb4043dfddaa">

  We reported the result using ```classification_report``` and plotted the linear graph for the training and validation loss of the model we built to see whether the model was overfitting/underfitting.

##### FIX BELOW
  
  It turns out a fair accuracy and the diagram shown below:
<img width="632" alt="截屏2024-03-14 17 59 17" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/a065caa2-7af9-4242-a78f-21a66eca88bf">
  
Finally, we applied oversampling, specifically SMOTE, to our train datasets. We then trained the best model from hyperparameter tuning using the oversampling data. Using the fitted model, we reported the resulting accuracy using ```classification_report``` and plotted the graph for the training and validation loss of the model to check for overfitting/underfitting.

##### FIX BELOW

  The decreased accuracy: 
  <img width="491" alt="截屏2024-03-14 17 59 55" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c8496de3-3381-4072-b4bd-31ebfb36d64c">
  And crazy diagram:
<img width="611" alt="截屏2024-03-14 18 00 25" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/4ffb68b8-b8a0-4271-9ce4-23903a450315">


### Third Model
#### SVM

  For the third model, we tried to implement the SVM model, since SVM supports both numerical and categorical data, so it was more flexible in data processing compared to Naive Bayes. First, we did the data processing for our target using one-hot encoding as we did on the initial data processing part. We also encoded 'race' and 'sex' by one-hot encoding and manually separating the values. Then we scaled our data, using both ```MinMaxScaler()``` and ```StandardScaler()```. 
  
  The first time we tried ```MinMaxScaler()``` after implementing the SVM model, we found the accuracy was low, so we then used ```StandardScaler()```. However, ```StandardScaler()``` needs to reprocess the data, since we had one-hot encoded features, and ```StandardScaler()``` is only used for numerical data. To separate the numerical data from the categorical data, we utilized ```iloc[:, 19:]``` to extract the categorical encoded features, and dropped the ```'sex'``` feature. After getting the scaled X, we combined them using np.concatenate() methodto get the complete X_scaled. Before we passed the data to the SVM model, we split the data with a ratio of train:test = 80:20 using ```train_test_split```. 
  
  Then we tried our first SVM model, with set parameters ```C = 0.1``` , ```degree = 2```, and ```kernel ='poly'```. When we fit our SVM model, we needed to transform ```y_train``` to a 1-D array, so we passed ```y_train.idxmax(axis=1).values``` as the parameters. We also used ```.idxmax(axis=1).values``` to get ```t_true``` in order to pass the data to ```classification_report```. However, the result of our SVM model was a low accuracy score, so to improve the accuracy, we decided to try different parameters to build our SVM model. The second time, we used ```GridSearch``` to find optimized parameters. We set our grid as: ```param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}```.

1. Use of the ```'poly'``` kernel:

<img width="550" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/f4471a67-0176-42f1-8351-25ac5fed92ff">

2. Hyperparameter tuning using ```GridSearch```: 

<img width="770" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/f00b8140-44f6-4668-a84c-61f712ddda55">

We also used oversampling with ```SMOTE``` and ```RandomOverSampler```, but it also resulted in a low accuracy, as shown in the results section.

1. Use of ```SMOTE```:

<img width="813" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/60bd08ee-299b-411f-9b0d-7ccbccefd86c">

2. Use of ```RandomOverSampler```: 

<img width="813" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/f5caa2be-7c8e-47cc-85da-e29970c8bad0">

#### Decision Tree Learning
  Another model we tried for the third model was the ```Decision Tree Learning``` model.

  The data preprocessing was the same as processing for SVM, where we applied encoding and ```StandardScaler``` to make our data clean, but this time we tried ```XgBoost``` model, setting the parameters as follows: 
<img width="815" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/7332c03b-0863-433a-ac35-55cd5608e167">


We then decided to try different parameters as well to improve the accuracy by ```RandomizedSearchCV```. As the diagram shows below, we selected four parameters: ```'max_depth'```, ```'learning_rate'```, ```'n_estimators'```, and ```'subsample'```:
  <img width="398" alt="截屏2024-03-14 17 07 35" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/4a03bae0-649e-4da1-9f6d-dead4ca3a3a3">

  After getting the best parameter, we rebuilt the model and finally got good accuracy which is 0.58, the result is displayed below:
  <img width="805" alt="截屏2024-03-14 17 08 48" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/68ed1809-e36c-4605-98f1-33eb56a80a72">

  We also tried another parameter search method, which is ```GridSearchCV```, in order to get the best model parameter.
In order to minimize the error, we set the same parameter as ```RandomizedSearchCV```, and finally we got the result: 
<img width="832" alt="截屏2024-03-14 17 11 10" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/69580227-0f0c-415c-b913-431d657bc45a">

  The accuracy increased again! It is worth trying different tuner methods! As the result shows, we decided to use the result from the ```GridSearchCV``` to do the ```classification report```: 
  <img width="434" alt="截屏2024-03-14 17 12 50" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/42b27107-5c23-4672-80ae-40052f7f2c76">

  Although it turns out the accuracy is a little bit decreased due to some training issues that are out of our control, we think that is the best accuracy we can get so far.

  We also printed the ranking of the importance of each feature: 
<img width="717" alt="截屏2024-03-15 18 27 35" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/57e56250-556b-4db5-8a26-fa4443c9cc93">

### Gradient boosted Tree
Gradient boosted Tree is the third method we chose.
Same thing as before, we manually tried one set of parameters that we think is worth trying, we set parameters as ```n_estimators=100, learning_rate=0.1, max_depth=3, random_state=21```, after we print the output of the ```classification report```, it gives us a really high accuracy, it is 0.56! (compare to the previous results):
<img width="456" alt="截屏2024-03-14 17 21 33" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c6087504-a4cf-4818-9333-3fc70bab96c1">

  After getting high results, we also tried OverSamplying, trying to make the number of our data balance, in that case, we applied ```SMOTE```, 
  


### KNN Model
KNN is our last try for the third model. We simply applied ```KNeighborsClassifier``` function, and below is the detail about our parameters in our KNN model: 
```
#KNN. similar result to SVM
k = 10
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)

# Predict the labels for the test set
y_true = y_test
y_pred = knn_classifier.predict(X_test)
```


# Results : 

### Model 1

### Model 2

### Model 3

#### SVM

- Evaluation after ```SMOTE```:
<img width="580" alt="截屏2024-03-14 17 33 01" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/6610ab7f-50ac-4a1f-a002-ab08c690bd06">

   - Evaluation after ```RandomOverSampler```:
<img width="569" alt="截屏2024-03-14 17 33 57" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c5e7741f-5925-427d-a3aa-2639c6649272">

#### Decision Tree Learning

### Gradient boosted Tree

After OverSamplying, then finally we found that the accuracy decreased:: <img width="439" alt="截屏2024-03-14 17 28 22" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/75220e47-7dab-40ce-9f42-89c85be67dbc">

### KNN Model

We printed the result using ```classification_report``` for our KNN model:
<img width="504" alt="截屏2024-03-14 17 39 16" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/5ba49df3-f499-47c9-af02-abac6b840a47">






# Discussion: 
### How we choose the model:

### EDA

Our objective was to find a populous dataset with multiple indicators of health as well as firm confirmations of the existence of diseases based on those indicators. Therefore, the SUPPORT2 dataset looked exactly like what we were searching for. It had all of the indicators, like ```’age’```, ```’sex’```, ```’race’```, ```’heart rate’```, ```’respiration rate’```, ```’temperature’```, and more that we could use for our targets during model construction and training. However, through further inspection at some of the classes, we found that some had missing values, with some columns having far more null values than complete values. 

Similarly, through our EDA, we found that the number of target classes within ```‘dzgroup’``` were imbalanced. Some of the target diseases appeared much more than others, which we knew could lead to some issues in the future and as a result we might have to perform some oversampling in order to allow our models to better learn from our data.
![Barchart](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/66afc575-09cd-4fc2-b152-805c58aabf06)



### Data Preprocessing

  For the preprocessing step, our goal was to properly partition the dataset into two groups: useful data and extraneous information. We decided to fixate on the following columns, finding that they were more relevant for our model and project goal: ```'age'```, ```'sex'```, ```'dzgroup'```, ```'scoma'```, ```'race'```, ```'sps'```, ```'aps'```, ```'diabetes'```, ```'dementia'```, ```'ca'```, ```'meanbp'```, ```'wblc'```, ```'hrt'```, ```'resp'```, ```'temp'```, ```'pafi'```, ```'alb'```, ```'bili'```, ```'crea'```, ```'sod'```, and ```'ph'```. We then tried to isolate the target values that we thought would be the best for results, and so we focused on the column ```‘dzgroup’```, since it contained important information like rate of colon cancer, coma, lung cancer, and more, all of which fell under the targets we were looking for. Finally, after securing our features and targets, we looked to make the entire dataset all readable data, so we dropped every row of data containing a null value to ensure the data was properly aligned and evenly spread across all features and targets. At the point of completing preprocessing, we were satisfied with the resulting dataset we got, as there were still plenty of entries to properly train each model. However, looking back now, maybe it would have been better to keep some of the null values, since it would have been better at training models even at the expense of exposing it to null values that could mess up the training.

### After Data Preprocessing:

  We chose to use a ```multi-class logistic regression model``` to train and fit the model with the preprocessed data. We then used the results to test for overfitting vs underfitting, accuracy, and error for train vs test data. We are thinking of testing with a ```multi-class classification model``` and a ```Keras Sequential model``` next to look for better results. This is because we need models that are capable of outputting multiple different classes since our targets are multiple different diseases. These next two models should be more powerful and hopefully better at predicting our targets.
  
  In conclusion, we thoroughly analyzed the description of each variable to understand their significance, carefully selecting features pertinent to our topic to ensure effective data preprocessing. We meticulously printed out the data frame containing the chosen features, meticulously verifying its shape to ascertain the number of observations accurately. We conducted a meticulous check for any empty values, such as null, within the data frame, ensuring data integrity before proceeding. Additionally, for each categorical attribute, we meticulously listed out all unique elements, ensuring comprehensive understanding and meticulous preprocessing of the data.

### Model 1:
  For our Model1, our goal is to build a baseline model that can help us understand the dataset better, and serve as comparison for our more complex future models. We decided to implement a logistic classifier for its ease of implementation and high interpretability. By default, logistic classifiers are designed for binary classification. However, using the parameter multi_class='multinomial', they are able to handle multi-class classification.

  Our Model1 achieved 0.55 training accuracy and 0.54 testing accuracy, with the cross-validation score also being very similar to the training score. Based on these observations, we concluded that no overfitting occurred. 

  Although the accuracy is not great, at this point, we were satisfied with Model 1 and its performance. We believed that the low accuracy was mainly due to the poor choice of model as a logistic classifier is a binary classification algorithm at the end of the day. With more advanced models and careful tuning, we should be able to level up that accuracy in the future.

### Model2

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

  In the meantime, we also tried Gradient Boosted Tree as an alternative to XGboost, but the results were not as good. We also displayed the ranking of importance of each feature, results shown below:
  <img width="717" alt="截屏2024-03-15 18 27 35" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/2bc722a5-6367-4467-939a-36a2c2b67c9b">

  As the picture shows, surprisingly, the ```race``` is the most important feature to discuss about towards our target.


  At this point, we have tried essentially every model besides KNN that has been discussed in this class. We ended up trying KNN as well. Our KNN model only yielded 0.5 testing accuracy, quite a bit worse than XGboost.



# Conclusion: 
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
   Contribution:
2. Name: Sebastian Diaz
   Contribution:
3. Name: Jou-Chih Chang
   Contribution:
4. Name: Juan Yin
   Contribution:
5. Name: Irving Zhao
   Contribution:
6. Name: Xianzhe Guo
   Contribution:
7. Name: Tiankuo Li
   Contribution:

# Colab Files of Our 3 Models
[Preprocessing](https://colab.research.google.com/drive/1nzW6bMa3XklLFByw_Fc9XWii4gMwAa67?usp=sharing)

[Model 1](https://colab.research.google.com/drive/1PFt7mk4PJi3zmMCn9rKbJEDA61fFsDZk?usp=sharing)

[Model 2](https://colab.research.google.com/drive/1GBM_WtSZDAe_ifttldpFGWvh5ttAP842?usp=sharing)

[Model 3](https://colab.research.google.com/drive/1X1-l40jQnPeq46wu0CgNISCQnN9mFe1C?usp=sharing)
