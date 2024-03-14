# CSE 151A Project

# Abstract about the course and our project

# Introduction 
(about our data and project: Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?)

# Methods: 
#### Data Preprocessing: 
  For data exploration, we called [pd.DataFrame()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=RA1zgeeIgR2p&line=3&uniqifier=1) function to display the variable name and descriptions, and we found that there are 47 variables in total, which is a large dataset.
  Then we created a data frame using [read_csv](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=YC0o6IQ1i-ec&line=1&uniqifier=1), after transferring, we selected 21 features that we think worth to work with, they are: 'age','sex','dzgroup','scoma','race','sps','aps','diabetes','dementia','ca','meanbp','wblc','hrt','resp','temp','pafi','alb','bili','crea','sod','ph'
  The next step is checking for null. We used [df.isnull().sum()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=KtldNNfpP723&line=1&uniqifier=1) function to calculate null value for each feature we selected.
  The result of df.isnull().sum() shows some features contain a bunch of null values, we are considering how to deal with this data, we will either completely remove them or refill them with mean/median data depending on the number of null values for that feature.
  For the features containing value type string, we applied [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to distinguish.
  For the binary feature like "sex", we divided them into integers 0 and 1 using the function :
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
  For nonbinary features, we applied [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function
  After the above data exploration and preprocessing, we are able to apply some visualization tools to help us explore the pattern of data
##### Data Preprocessing: 
2. We print out a correlation matrix plot of the data frame in the form of a heatmap.
3. We print out the count of the unique elements in the 'dzgroup' column in the form of a bar plot.
4. We one hot encode all of the categorical attributes.
5. After one hot encoding, we drop all of the original categorical attributes and all of the empty values.
6. We check again to see whether there is still any empty value in the data frame.

#### Visualization Tool
1.Parallel Coordinates Plot: we applied this function to visualize the relationship between "dementia" and other features.

2. Plotly Express interface: This function is used to observe the relationship between "age" and dimensions=['age', 'sps', 'meanbp','scoma','aps'], it seems like people's physical features change around 50.

3. Multiple Line Plots: this function is used to check the pattern of two particular features.
   - first, we applied an age-diabetes pair, which shows people between 40 to 80 are the main group to have diabetes
   - second, we applied the bili-hrt pair and bili-ph pair, their diagram has a similar pattern, and We think we should implement more data to see the pattern between them.

4. We apply the Pairplot for the entire dataset twice, before and after we split the data using one-hot encoding.


### multi-class logistic regression - First model:
  In order to make it easy to observe some patterns, we dicide to apply multi-class logistic regression first.
  
  The first model is not the most precise, as there is a relatively clear sign of overfitting due to the cross-validation score being lower in the start than the training score, and only a relative evening out towards the end of the graph. We can possibly improve this model by selecting different features for training use to further finetune the results and not have overfitting or underfitting for the model.
  
  In our first model, we set ```'dzgroup'``` as our target and used the rest of the columns as our features. To ensure that our feature data was normalized, we implemented minmax normalization. We then split the data into training and testing sets, with a 70:30 ratio and a random state of 0. We built eight different logistic regression models to predict each target and reported the results using ```accuracy```, ```classification report```, and ```confusion matrix```. We also generated ```learning curves``` for each logistic regression model, calculating mean training and testing scores across different ```cross-validation folds``` for each training and testing size.   
  In addition, we built a logistic regression model that predicts multiclass('dzgroup') and reported the results using ```accuracy```, ```classification report```, and ```confusion matrix```. We also generated learning curves for the logistic regression model, calculating mean training and testing scores across different cross-validation folds for each training and testing size. Finally, We plot learning curves for the logistic regression model.
  
  Overall, our analysis was thorough and rigorous, ensuring that our results were accurate and reliable.

#### First Model Visualization 
![Screenshot 2024-03-10 at 7 16 17 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/90e3f1ea-bb2b-4f66-ad46-c7322d1a7d89)

This is our plot for our multi-class regression model, comparing our training score vs our cross validation score.
This plot shows that there is some underfitting in our model and thus a logistic regression model likely isn't the best model we could use for our data, which is useful to know for our future models. We also had other similar plots for our logistic regression models that we did for each target rather than multiple at once.

![Screenshot 2024-03-10 at 7 20 21 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/f9e92ef8-55ad-4b95-ad17-ad58568ef99d)

Here is one of those models, the model itself doesn't look too different from the other models for each target and the results are relatively the same.



### SVM - Third model
#### First Try

  For the third model, we tried to implement the SVM model, since SVM supports both numerical and categorical data, it is more flexible in data processing compared to Naive Bayes. First we did the data processing for our target using one-hot encoding as we did on the initial data processing part. We also encoded 'race' and 'sex' by one-hone encoding and manually separating. Then we try to scale our data, we were think about using the ```MinMaxScaler()``` or ```StandardScaler()```, since both are fine, we decide to try both. 
  
  The first time we tried minimax, after implementing the SVM model, we found the accuracy was low, so we went back to the second Scaler method, which is the StandardScaler. However, the StandardScaler needs to reprocess the data, since we have some one-hot encoded features, and StandardScaler is only used to process the numeric data, we need to separate the Numeric data and categorical data. In that case, we utilized ```iloc[:, 19:]``` to extract the categorical encoded features, and drops the 'sex' feature. After getting the scaled X, we combine them using np.concatenate() method, so now we have the complete X_scaled. Before we pass the data to the SVM model, we split data to train:test = 80:20. 
  
  Then we tried our first SVM model, with set parameters ```C = 0.1``` , ```degree = 2```, and ```kernel ='poly'```. When we fit our SVM model, we need to transform our y_train to 1-D array, so we passed ```y_train.idxmax(axis=1).values``` as my parameters. We also used ```.idxmax(axis=1).values``` to get our t_true in order to pass the data to the ```classification_report```. And we can observe the result of our SVM model. In order to improve the accuracy, we decided to try different parameters to build our SVM model. So second time we try to use hyper-parameter-grid to find optimized parameters. We set our grid as: ```param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}```.

#### Second try


# Results : 

# Discussion: 
### How we choose the model:
After data preprocessing: #### Summary: 
  We chose to use a ```multi-class logistic regression model``` to train and fit the model with the preprocessed data. We then used the results to test for overfitting vs underfitting, accuracy and error for train vs test data. We are thinking of testing with a ```multi-class classification model``` and a ```Keras Sequential model``` next to look for better results. This is because we need models which are capable of outputing multiple different classes since our targets are multiple different diseases. These next two models should be more powerful and hopefully better at predicting our targets.
  In conclution, we thoroughly analyzed the description of each variable to understand their significance, carefully selecting features pertinent to our topic to ensure effective data preprocessing. We meticulously printed out the data frame containing the chosen features, meticulously verifying its shape to ascertain the number of observations accurately. We conducted a meticulous check for any empty values, such as null, within the data frame, ensuring data integrity before proceeding. Additionally, for each categorical attribute, we meticulously listed out all unique elements, ensuring comprehensive understanding and meticulous preprocessing of the data.

After getting result for model 1:


# Conclusion section: 

# Collaboration section: 
