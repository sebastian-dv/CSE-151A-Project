# CSE 151A Project

# Abstract about the course and our project

# Introduction 

This project aims to investigate and identify key factors that contribute to the diagnosis of specific diseases. Predicting the presence of various diseases based on a single factor is challenging; therefore, we have endeavored to predict them using multiple features available through machine learning techniques. This project will be carried out using the SUPPORT2 dataset from the UC Irvine Machine Learning Repository. This dataset comprises records of 9,105 critically ill patients from five United States medical centers, collected between 1989-1991 and 1992-1994, as illustrated by [Figure 1](#figure-1-support-project-phases). Each row contains the records of hospitalized patients who met the inclusion and exclusion criteria for one of eight disease categories: acute respiratory failure, chronic obstructive pulmonary disease, congestive heart failure, coma, colon cancer, lung cancer, multiple organ system failure with malignancy, and multiple organ system failure with sepsis. For disease prediction, we selected 21 features from this dataset, such as physiology scores, chemical levels, and various measurements on Day 3. [Table 1](#table-1-uci-dataset-features-detailed-information) presents the names and descriptions of these features. Machine Learning algorithms—logistic regression, neural networks, decision trees with XGBoost and Gradient Boosting, Support Vector Machines, and K-Nearest Neighbors—are employed, and we will later examine how our results fare in predicting these diseases based on the available data.


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
  Then we created a data frame using [read_csv](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=YC0o6IQ1i-ec&line=1&uniqifier=1), after transferring, we selected 21 features to work with: ```'age'```, ```'sex'```, ```'dzgroup'```, ```'scoma'```, ```'race'```, ```'sps'```, ```'aps'```, ```'diabetes'```, ```'dementia'```, ```'ca'```, ```'meanbp'```, ```'wblc'```, ```'hrt'```, ```'resp'```, ```'temp'```, ```'pafi'```, ```'alb'```, ```'bili'```, ```'crea'```, ```'sod'```, and ```'ph'```.
  The next step we took was to check for null values in the dataset. We used the [df.isnull().sum()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=KtldNNfpP723&line=1&uniqifier=1) function to calculate the null values for each feature we selected.
  The result of df.isnull().sum() showed that some features contained several null values, and so we decided to deal with this data by either completely removing them or refilling them with mean/median data depending on the number of null values found for that feature.
  For the features containing value type string, we applied the [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to distinguish.
  For binary features like ```'sex'```, we divided them into integers 0 and 1 using the function
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
  For nonbinary features, we applied the [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function to get them in a binary format.
  After the above data exploration and preprocessing, we were able to apply some visualization tools to help us explore the pattern of data.
##### Data Preprocessing: 
2. We printed out a correlation matrix plot of the data frame in the form of a heatmap.
3. We printed out the count of the unique elements in the ```'dzgroup'``` column in the form of a bar plot.
4. We one-hot encoded all of the categorical attributes.
5. After one-hot encoding, we dropped all of the original categorical attributes and all of the empty values.

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
We plotted the linear graph for the training and validation loss of the model we built to see the performance of the model as well as if it was overfitting/underfitting:

### REPEATING TEXT (Anything to keep from this?) - START

Next, we built another model (Keras Sequential Model) to predict each target and report the result using average accuracy. We design a 5-layer artificial neural network using the Tanh activation function in each layer and the softmax activation function in the output layer. The Number of nodes in the first layer is 72, the number of nodes in the output layer is 8, and the number of nodes in the rest of the layer is 42. When we compile the model, we use Stochastic Gradient Descent to minimize the error rate, use categorical cross entropy as the loss function, and specific ```MSE``` and ```accuracy``` as the metrics. When we fit the mode, we set the number of epochs to 50, the batch size to 20, and the verbose to 0. 

### REPEATING TEXT - END

  We performed K-Fold Cross-Validation with different random splits of the data. The entire cross-validation was repeated 5 times and in each repetition, the data was split into 10 subsets. 
  We calculated the accuracy for each subset of the data when processing cross-validation. Then, took the mean of it to see the performance of our model. The model and results are shown below:
  <img width="796" alt="截屏2024-03-14 18 03 19" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/b924ae2b-fda3-4a52-91ea-178df74ede69">

  Then, we built another model(hyperparameter tuning model) to predict each target and report the result using classification report. We set the range of units in the input layer to 18180, the range of units in hidden layers to 12180, and the units in the output layer to 8. The available choices for the activation function include ```'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu', and 'mish'```. The available choices for optimizer include ```'sgd', 'adam', and 'rmsprop.'``` The range of learning rate is ```0.001~0.1```. The available choices for the loss function include 'categorical_crossentropy', 'mse', and 'binary_crossentropy.' After that, we run the model to find the most optimized parameters and use them to rebuild our model.
  the hyperparameter tuning model was set up as below:
  <img width="904" alt="截屏2024-03-14 17 58 18" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/292037a1-333c-47e1-9235-cb4043dfddaa">

  We report the result using a classification report and plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting. It turns out a fair accuracy and the diagram shown below:
<img width="632" alt="截屏2024-03-14 17 59 17" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/a065caa2-7af9-4242-a78f-21a66eca88bf">
  
  The last thing we do is we apply OverSamplying (SMOTE) to our train datasets. Then, applying the oversampling data to our best model from our hyperparameter tuning. Then, report the result using a classification report and plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting.
  The decreased accuracy: 
  <img width="491" alt="截屏2024-03-14 17 59 55" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c8496de3-3381-4072-b4bd-31ebfb36d64c">
  And crazy diagram:
<img width="611" alt="截屏2024-03-14 18 00 25" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/4ffb68b8-b8a0-4271-9ce4-23903a450315">


### Third Model
#### SVM

  For the third model, we tried to implement the SVM model, since SVM supports both numerical and categorical data, it is more flexible in data processing compared to Naive Bayes. First, we did the data processing for our target using one-hot encoding as we did on the initial data processing part. We also encoded 'race' and 'sex' by one-hone encoding and manually separating. Then we tried to scale our data, and we were thinking about using the ```MinMaxScaler()``` or ```StandardScaler()```, since both are fine, we decided to try both. 
  
  The first time we tried minimax, after implementing the SVM model, we found the accuracy was low, so we went back to the second Scaler method, which is the StandardScaler. However, the StandardScaler needs to reprocess the data, since we have some one-hot encoded features, and StandardScaler is only used to process the numeric data, we need to separate the Numeric data and categorical data. In that case, we utilized ```iloc[:, 19:]``` to extract the categorical encoded features, and drop the 'sex' feature. After getting the scaled X, we combine them using np.concatenate() method, so now we have the complete X_scaled. Before we pass the data to the SVM model, we split the data to train:test = 80:20. 
  
  Then we tried our first SVM model, with set parameters ```C = 0.1``` , ```degree = 2```, and ```kernel ='poly'```. When we fit our SVM model, we need to transform our y_train to a 1-D array, so we passed ```y_train.idxmax(axis=1).values``` as my parameters. We also used ```.idxmax(axis=1).values``` to get our t_true in order to pass the data to the ```classification_report```. And we can observe the result of our SVM model. In order to improve the accuracy, we decided to try different parameters to build our SVM model. So second time we try to use hyper-parameter-grid to find optimized parameters. We set our grid as: ```param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}``` as follows: 
  <img width="765" alt="截屏2024-03-14 17 03 19" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/451bc212-dea4-4ec1-a14e-1fedaf738c0b">

  We also applied OverSamplying using ```SMOTE``` and ```RandomOverSampler```, but it turns out decreased the accuracy: 
   - Evaluation after ```SMOTE```:

<img width="580" alt="截屏2024-03-14 17 33 01" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/6610ab7f-50ac-4a1f-a002-ab08c690bd06">

   - Evaluation after ```RandomOverSampler```:
<img width="569" alt="截屏2024-03-14 17 33 57" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c5e7741f-5925-427d-a3aa-2639c6649272">

  We think 0.53 should be the best accuracy we can get from the SVM model.


#### Decision Tree Learning
  Another model we tried for the third model is ```Decision Tree Learning``` model.

  The data preprocessing is the same, we applied encoding and StandardScaler to make our data clean, but this time we tried ```XgBoost``` model, we set our parameters as follows: 
  <img width="830" alt="截屏2024-03-14 17 01 28" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/5621a169-42e4-4b7f-a4c4-9ac41731245d">

  It turns out a not-bad result, with 0.57-ish accuracy, and better than the tunered SVM model. And we decided to try different parameters as well to improve the accuracy by ```RandomizedSearchCV```. As the diagram shows below, we selected four parameters: ```'max_depth'```, ```'learning_rate'```, ```'n_estimators'```, and ```'subsample'```:
  <img width="398" alt="截屏2024-03-14 17 07 35" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/4a03bae0-649e-4da1-9f6d-dead4ca3a3a3">

  After getting the best parameter, we rebuilt the model and finally got good accuracy which is 0.58, the result is displayed below:
  <img width="805" alt="截屏2024-03-14 17 08 48" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/68ed1809-e36c-4605-98f1-33eb56a80a72">

  We also tried another parameter search method, which is ```GridSearchCV```, in order to get the best model parameter.
In order to minimize the error, we set the same parameter as ```RandomizedSearchCV```, and finally we got the result: 
<img width="832" alt="截屏2024-03-14 17 11 10" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/69580227-0f0c-415c-b913-431d657bc45a">

  The accuracy increased again! It is worth trying different tuner methods! As the result shows, we decided to use the result from the ```GridSearchCV``` to do the ```classification report```: 
  <img width="434" alt="截屏2024-03-14 17 12 50" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/42b27107-5c23-4672-80ae-40052f7f2c76">
  
  Although it turns out the accuracy is a little bit decreased due to some training issues that are out of our control, we think that is the best accuracy we can get so far.

### Gradient boosted Tree
Gradient boosted Tree is the third method we chose.
Same thing as before, we manually tried one set of parameters that we think is worth trying, we set parameters as ```n_estimators=100, learning_rate=0.1, max_depth=3, random_state=21```, after we print the output of the ```classification report```, it gives us a really high accuracy, it is 0.56! (compare to the previous results):
<img width="456" alt="截屏2024-03-14 17 21 33" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/c6087504-a4cf-4818-9333-3fc70bab96c1">

  After getting high results, we also tried OverSamplying, trying to make the number of our data balance, in that case, we applied ```SMOTE```, then finally we found that the accuracy decreased: 
  <img width="439" alt="截屏2024-03-14 17 28 22" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/75220e47-7dab-40ce-9f42-89c85be67dbc">


### KNN Model
KNN is our last try for the third model.

We simply applied ```KNeighborsClassifier``` function, and printed the result using ```classification_report```:
<img width="504" alt="截屏2024-03-14 17 39 16" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/79886525/5ba49df3-f499-47c9-af02-abac6b840a47">


# Results : 

# Discussion: 
### How we choose the model:
After data preprocessing:

  We chose to use a ```multi-class logistic regression model``` to train and fit the model with the preprocessed data. We then used the results to test for overfitting vs underfitting, accuracy, and error for train vs test data. We are thinking of testing with a ```multi-class classification model``` and a ```Keras Sequential model``` next to look for better results. This is because we need models that are capable of outputting multiple different classes since our targets are multiple different diseases. These next two models should be more powerful and hopefully better at predicting our targets.
  
  In conclusion, we thoroughly analyzed the description of each variable to understand their significance, carefully selecting features pertinent to our topic to ensure effective data preprocessing. We meticulously printed out the data frame containing the chosen features, meticulously verifying its shape to ascertain the number of observations accurately. We conducted a meticulous check for any empty values, such as null, within the data frame, ensuring data integrity before proceeding. Additionally, for each categorical attribute, we meticulously listed out all unique elements, ensuring comprehensive understanding and meticulous preprocessing of the data.

After getting the result for model 1:



# Conclusion section: 

# Collaboration section: 
