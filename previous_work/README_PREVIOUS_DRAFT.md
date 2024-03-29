# CSE-151A-Project

## Requirements:
### Data Exploration
- Read up on what the data means (Explain what the data is in a text cell)
- Visualize the data (display the DataFrame for features and targets)
- Look at a subsample of observations (Slice the data into training and testing sets? Remove extraneous features that add no value)
- Be one with the data 
### Preprocess Data
- Imputation (?)
- Normalization vs standardization
- Data Encoding: label encoding vs one hot encoding
- Data transformation

## Our Work
1. For data exploration, we called [pd.DataFrame()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=RA1zgeeIgR2p&line=3&uniqifier=1) function to display the variable name and descriptions, and we found that there are 47 variables in total, which is a large dataset.
2. Then we created a data frame using [read_csv](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=YC0o6IQ1i-ec&line=1&uniqifier=1), after transferring, we selected 21 features that we think worth to work with, they are: 'age','sex','dzgroup','scoma','race','sps','aps','diabetes','dementia','ca','meanbp','wblc','hrt','resp','temp','pafi','alb','bili','crea','sod','ph'
3. The next step is checking for null. We used [df.isnull().sum()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=KtldNNfpP723&line=1&uniqifier=1) function to calculate null value for each feature we selected.
4. The result of df.isnull().sum() shows some features contain a bunch of null values, we are considering how to deal with this data, we will either completely remove them or refill them with mean/median data depending on the number of null values for that feature.
5. For the features containing value type string, we applied [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to distinguish.
6. For the binary feature like "sex", we divided them into integers 0 and 1 using the function :
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
8. For nonbinary features, we applied [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function
9. After the above data exploration and preprocessing, we are able to apply some visualization tools to help us explore the pattern of data
10. We then chose to use a multi-class logistic regression model to train and fit the model with the preprocessed data. We then used the results to test for overfitting vs underfitting, accuracy and error for train vs test data. We are thinking of testing with a multi-class classification model and a Keras Sequential model next to look for better results. This is because we need models which are capable of outputing multiple different classes since our targets are multiple different diseases. These next two models should be more powerful and hopefully better at predicting our targets.


## Visualization Tool
1. Parallel Coordinates Plot: we applied this function to visualize the relationship between "dementia" and other features.
2. Plotly Express interface: This function is used to observe the relationship between "age" and dimensions=['age', 'sps', 'meanbp','scoma','aps'], it seems like people's physical features change around 50.
3. Multiple Line Plots: this function is used to check the pattern of two particular features.
   - first, we applied an age-diabetes pair, which shows people between 40 to 80 are the main group to have diabetes
   - second, we applied the bili-hrt pair and bili-ph pair, their diagram has a similar pattern, and We think we should implement more data to see the pattern between them.
4. We apply the Pairplot for the entire dataset twice, before and after we split the data using one-hot encoding.

### First Model
The first model is not the most precise, as there is a relatively clear sign of overfitting due to the cross-validation score being lower in the start than the training score, and only a relative evening out towards the end of the graph. We can possibly improve this model by selecting different features for training use to further finetune the results and not have overfitting or underfitting for the model.

### Visualization 
![Screenshot 2024-03-10 at 7 16 17 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/90e3f1ea-bb2b-4f66-ad46-c7322d1a7d89)
This is our plot for our multi-class regression model, comparing our training score vs our cross validation score.
This plot shows that there is some underfitting in our model and thus a logistic regression model likely isn't the best model we could use for our data, which is useful to know for our future models. We also had other similar plots for our logistic regression models that we did for each target rather than multiple at once.

![Screenshot 2024-03-10 at 7 20 21 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/f9e92ef8-55ad-4b95-ad17-ad58568ef99d)
Here is one of those models, the model itself doesn't look too different from the other models for each target and the results are relatively the same.

## Second Model
## Requirements:
In this milestone you will focus on building your second model. You will also need to evaluate your this model and see where it fits in the underfitting/overfitting graph.

1. Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.

2. Train your second model

3. Evaluate your model compare training vs test error

4. Where does your model fit in the fitting graph, how does it compare to your first model?

5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

5. What is the plan for the next model you are thinking of and why?

6. Update your readme with this info added to the readme with links to the jupyter notebook!

7. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform compared to your first and why?

## Our Work
Our data and labels were sufficient for the most part, we didn't have to change our method of preprocessing for this model from the preprocessing in our first. We still scaled our numerical data as well as one hot encoded our categorical data. For our loss function we were able to find the most optimal one through hyperparameter tuning.

Our Second model is a Neural Network model.

First, we tried manually creating a 5-layer model (including input and output layers) with the 'tanh' activation function for the input and hidden layers, and the 'softmax' activation function for the output layer since we got encoded target y. We observed the results with a classification report and plot which simulate the distance between our testing result and training input data as seen below in our visualizations.

After testing with the first model, the accuracy was low, at ~0.56. We first decided to apply K-fold Cross-validation, to get a more reliable estimate of how our model will perform on unseen data. After running K-fold, the average of all folds was at ~0.53 accuracy. Although the accuracy was consistent with our result from the test set, it was still very low. We decided to improve our accuracy through other methods like K-fold and hyperparameter tuning.

Then we built a hyperparameter tuning model by tuning the units, activation function, optimizers, learning rates, and loss functions, trying to find the most optimized parameters to rebuild our model, we sort our tuner result by the score of accuracy. After finding the best model parameters, we rebuilt our model, printed the classification report, and displayed the plot again. This time, it turns out our accuracy actually didn't improve by much, it was still very similar to the accuracy we achieved with our base model and with K-fold, at ~0.51. So we continued looking for a way to achieve a higher accuracy. 
Then we decided to apply OverSampling since we found that our target classes were imbalanced. After applying the RandomOverSampler to our best model from our hyperparameter tuning, we got 0.76 accuracy.

## Visualization
### Manually Created 5-Layer Model 
![image](https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/009f4f31-a8a0-4e4f-8cc9-b0e4c10e095c)
<img width="434" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/311efdfb-e9a9-4c78-b251-e838a7ed7c99">

This graph shows signs of overfitting, as seen by the disparity in loss apparent throughout the entire graph, which is different from the first model, as the first model was underfitting.
### Hyperparameter Tuned Model
![image](https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/47181271-be7a-444c-a1aa-655d1195fef1)
<img width="429" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/d7a470d5-3f18-4cc8-ab90-dc155713a87e">

Similar, to the initial version of the second model, this graph shows signs of overfitting, as seen by the disparity in loss apparent throughout the entire graph, which is different from the first model, as the first model was underfitting.
### Hyperparameter Tuned Model with Oversampling
![image](https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/3a5a0026-0896-4357-b8fb-7e49b2f5521d)
<img width="425" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/293e2b68-8e5f-4c04-a968-e003df5293c8">

This graph shows no major signs of underfitting or overfitting, as the losses are closely tied to each other, and progress at a similar rate, which is far better than the underfitting graph produced by the first model.

## Conclusion
Our second model has proved to be more accurate in predicting our target classes than our previous (multiclass) logistic regression model. Although, it started off very inaccurate and in some cases worse, we were able to apply different techniques such as K-fold, hyperparameter tuning, and oversampling, in order to improve the accuracy of our model. We believe that the largest contributor to this second model improving over the first was likely because of the oversampling we did. The oversampling helped balance out our target classes much more and helped our model learn much more efficiently. There is likely still room for improvement through feature engineering techniques, more complex hyperparameter tuning, and more time, but the improvement would probably only be very minimal over what we've achieved already. For our next model, we are thinking of doing an SVM. We decided on an SVM because of its ability to support both categorical and numerical features, as well as it's ability to do multiclass classification. We want to possibly try a couple of different kernels in order to find which works best for our data.

## Jupyter Notebook Link
[Link to Jupyter Notebook in Github](https://github.com/sebastian-dv/CSE-151A-Project/blob/main/Milestone%204%3A%20Model%202.ipynb)

[Link to Jupyter Notebook in Google Colab](https://colab.research.google.com/drive/1quxLTGDJ_VTTTDzPrHU4nsm8mzdlV4yg?usp=sharing)


# Model 3
## Model 3 Reqirements
Your 3rd model must undergo the same requirements as Model 1 and Model 2 for evaluation. Please see the previous milestone for clarification: 
1. Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.

2. Train your third model

3. Evaluate your model and compare training vs test error

4. Where does your model fit in the fitting graph, how does it compare to your first model?

5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

5. What is the plan for the next model you are thinking of and why?

6. Update your readme with this info added to the readme with links to the jupyter notebook!

7. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform compared to your first and why?

# Project Summary - Final Submission
You will require the following:

1. A complete Introduction
   For model 1, we used the logistic regression model.
   For model 2, we used neural network.
   For model 3, we used support vector machine.
3. A complete submission of all prior submissions
   [Link to Model 1]()
   [Link to Model 2]()
   [Link to Model 3]()
5. All code uploaded in the form of Jupiter notebooks that can be easily followed along to your GitHub repo
   [Link to our GitHub](https://github.com/sebastian-dv/CSE-151A-Project)
6. A completed write that includes the following:
#### Introduction of your project. Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?

#### Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.

#### Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, Model 3, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods
Results section. This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.
##### Data Exploration:
1. We take a look at the description of each variable.
2. We select the features that are related to our topic.
3. We print out the data frame of the features we selected.
4. We check the shape of the data frame. For example, the number of observations in our data frame.
5. We check whether there is any empty value(ex. null) in the data frame.
6. For each of the categorical attributes, we print out all the unique elements.
##### Data Preprocessing: 
1. If the value in the 'sex' column is 'female', we replace it with 0. Otherwise, replace it with 1(male).
2. We print out a correlation matrix plot of the data frame in the form of a heatmap.
3. We print out the count of the unique elements in the 'dzgroup' column in the form of a bar plot.
4. We one hot encode all of the categorical attributes.
5. After one hot encoding, we drop all of the original categorical attributes and all of the empty values.
6. We check again to see whether there is still any empty value in the data frame.
7. We set 'dzgroup' as our target and the rest of the columns are our features.
##### Model 1(logistic regression model):
   ##### Data Preprocessing:
   1. We implement minmax normalization to our feature data.
   2. We split the data into training and testing set by 70:30 and set the random state to 0.
   ##### Build Model and Report the Result:
   1. We build eight different logistic regression models that predict each target and report the result using accuracy, classification report, and confusion matrix.
   2. We generate learning curves for each logistic regression model and calculate mean training and testing scores across different cross-validation folds for each training and testing size.
   3. We plot learning curves for each logistic regression model.
   4. We also build a logistic regression model that predicts multiclass('dzgroup') and report the result using accuracy, classification report, and confusion matrix.
   5. We generate learning curves for the logistic regression model and calculate mean training and testing scores across different cross-validation folds for each training and testing size.
   6. We plot learning curves for the logistic regression model.
##### Model 2(neural network):
   ##### Data Preprocessing:
   1. We implement minmax normalization to our feature data.
   2. For Keras Sequential Model, we split the data into training and testing set by 80:20 and set the random state to 0. For hyperparameter tuning model, we split the data into training and testing set by 85:15 and set the random state to 1.
   ##### Build Model and Report the Result:
   1. We build the base model(Keras Sequential Model) to predict each target and report the result using classification report. We design a 5-layer artificial neural network using the Tanh activation function in each layer and the softmax activation function in the output layer. The Number of nodes in the first layer is 72, the number of nodes in the output layer is 8, and the number of nodes in the rest of the layer is 42. When we compile the model, we use Stochastic Gradient Descent to minimize the error rate and use categorical cross entropy as the loss function. When we fit the mode, we set the number of epochs to 100, the batch size to 20, the validation split to 0.1, and the verbose to 0.
   2. We plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting.
   3. Next, We build another model(Keras Sequential Model) to predict each target and report the result using average accuracy. We design a 5-layer artificial neural network using the Tanh activation function in each layer and the softmax activation function in the output layer. The Number of nodes in the first layer is 72, the number of nodes in the output layer is 8, and the number of nodes in the rest of the layer is 42. When we compile the model, we use Stochastic Gradient Descent to minimize the error rate, use categorical cross entropy as the loss function, and specific mse and accuracy as the metrics. When we fit the mode, we set the number of epochs to 50, the batch size to 20, and the verbose to 0.
   4. We perform repeat K-Fold Cross-Validation with different random splits of the data. The entire cross-validation will repeat 5 times and in each repetition, the data will be split into 10 subsets.
   5. We calculate the accuracy for each subset of the data when processing cross-validation. Then, take the mean of it to see the performance of our model.
   6. Then, we build another model(hyperparameter tuning model) to predict each target and report the result using classification report. We set the range of units in the input layer to 18~180, the range of units in hidden layers to 12~180, and the units in the output layer to 8. The available choices for the activation function include 'relu', 'sigmoid', 'tanh', 'softmax', 'linear', 'leaky_relu', and 'mish'. The available choices for optimizer include 'sgd', 'adam', and 'rmsprop.'  The range of learning rate is 0.001~0.1. The available choices for the loss function include 'categorical_crossentropy', 'mse', and 'binary_crossentropy.' After that, we run the model to find the most optimized parameters and use them to rebuild our model.
   7. We report the result using classification report and plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting.
   8. The last thing we do is we apply OverSampling(SMOTE) to our train datasets. Then, applying the oversampling data to our best model from our hyperparameter tuning. Then, report the result using classification report and plot the linear graph for the training and validation loss of the model we build to see whether the model is overfitting/underfitting.
##### Model 3():
   ##### Data Preprocessing:
   1.
   ##### Build Model and Report the Result:
   1.
##### Results:


#### Discussion section: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!
#### Data Preprocessing
   ##### Data Clean
   1. First, we try to figure out the meanings of each column of our data to choose our target, so we pulled out the description of each column and tried to find the pattern between some features and others.
   2. To do the data preprocessing, we tried to check how big our data is, if the dataset is too small, we will try to replace the null with 0 or another integer value; while if the dataset is big enough, we will drop the null without replacement.
      By applying ```print("Number of observations:", len(df))``` and ```df.shape```, It turns out we got a big data set with shape (9105, 22), so we just dropped the null values.
   3. After that, by looking at the dataset, we need to find the non-numeric data and encode these columns to make the model consistent data type.
   4. We tried to encode all of the categorical data using one-hot encoding, for example, features ```'dzgroup', 'race', 'ca'```. And for the binary features for example ```'sex'```, we manually encoded female to 0, and male to 1;
   5. After removing null and encoding, we decided to double-check our value and make sure our dataset was ready to be processed. We applied ```heatmap``` to observe the simulations 
   ##### Relation Simulations
   1. We had no idea how to choose our target, so we applied a pair plot, however, the plot is huge, and hard to observe the pattern between features 
   2. Then we tried randomly choosing a target: ```'dementia'```, and applied Parallel Coordinates Plot, the result shows there is just one feature that has a strong relationship with ```'dementia'```, Since that, we think use multiple features to predict ```'dementia'``` is not going to meaningful, so we decide to try other features.
   3. We also tried plot features ```'age'``` and ```'diabetes'``` using Multiple Line Plots, it turns out the plot is hard to observe as well. For the Multiple Line Plots, we also tried to plog the features ```'hrt'``` and ```'ph'```, but the result looks the same, and still hard to observe the patterns.
   4. So we were trying to do the observation using other clear models.
   5. First of all, we tried to find which feature has the most useful data, so we applied a heatmap, it turns out our feature is good enough to be processed.
   6. After discussing and observing the data, we decided to make  ```'dzgroup'``` our target, since the feature includes multiple diseases that are valuable to predict.
   7. After we decided on our target, the first thing to do was encode, we discussed which encoding method we should use, and discussed the future model we were going to use to process the data, and after discussion, we decided to encode our target using one-hot encoding first since we are plan to use the regression model as our first model, so one-hot encoding is more appropriate. And we could re-encoding our target by other encoding methods if we need it later.
   8. After encoding, we displayed the number of each entry for our encoded  ```'dzgroup'``` target. It turns out our data is not balanced, which means we might need to do the OverSamplying before using SVM for our future model.
 ##### Model 1:
 
 ##### Model 2:
 sequential model, 5 layers NN, tuner, 
 
 ##### Model 3:
 1. For the third model, we tried to implement the SVM model, since SVM supports both numerical and categorical data, it is more flexible in data processing compared to Naive Bayes.
 2. First we did the data processing for our target using one-hot encoding as we did on the initial data processing part. We also encoded ```'race'``` and ```'sex'``` by one-hone encoding and manually separating.
 3. Then we try to scale our data, we were think about using the ```MinMaxScaler()``` or ```StandardScaler()```, since both are fine, we decide to try both.
 4. the first time we tried minimax, after implementing the SVM model, we found the accuracy was low, so we Back to the second Scaler method, which is the StandardScaler.
 5. However, the StandardScaler needs to reprocess the data, since we have some one-hot encoded features, and StandardScaleris only used to process the numeric data, we need to separate the Numeric data and categorical data.
 6. In that case, we utilized ```iloc[:, 19:]``` to extract the categorical encoded features, and drops the ```'sex'``` feature.
 7. After getting the scaled X, we combine them using ```np.concatenate()``` method, so now we have the complete ```X_scaled```
 8. Before we pass the data to the SVM model, we split data to train:test = 80:20
 9. Then we tried our first SVM model, with set parameters ```C = 0.1``` , ```degree = 2```, and ```kernel ='poly'```.
 10. When we fit our SVM model, we need to transform our ```y_train``` to 1-D array, so we passed ```y_train.idxmax(axis=1).values``` as my parameters.
 11. We also used ```.idxmax(axis=1).values``` to get our t_true in order to pass the data to the ```classification_report```.
 12. And we can observe the result of our SVM model.
 14. In order to improve the accuracy, We decided to try different parameters to build our SVM model.
 15. So seconde time we try to use hyper-parameter-grid to find optemiazed parameters.
 16. We set our grid as: ```param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf', 'poly']}```
 17. We printed the accuracy again after using the hyper parameter, and it shows our model slightly improved our accuracy. And finally we got : ```Best Hyperparameters: {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}```
 18. We still think our accuracy is not high enpugh, so we decide to apply the OverSamplying as we did for HW4, and as we mentioned before, our data is imbalanced, so we think it's better to apply the overSamplying.
 19. We also tried two different OverSamplying methods this time, the first is SMOTE, and the second is RandomOverSampler.
 20. And we also trying to know wheathr scaling will helping the OverSamplying, so we have four different cases in total: SMOTE with scaled, SMOTE with unscaled, RandomOverSamplerwith scaled, and  RandomOverSampler with unscaled.
 21. We compare these four to find the best accuracy using the ```classification_report```
 
 
#### Conclusion section: This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts

#### Collaboration section: This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc. Please be truthful about this as this will determine individual grades in participation. There is no job that is better than the other. If you did no code but did the entire write up and gave feedback during the steps and collaborated then you would still get full credit. If you only coded but gave feedback on the write up and other things, then you still get full credit. If you managed everyone and the deadlines and setup meetings and communicated with teaching staff only then you get full credit. Every role is important as long as you collaborated and were integral to the completion of the project. If the person did nothing. they risk getting a big fat 0. Just like in any job, if you did nothing, you have the risk of getting fired. Teamwork is one of the most important qualities in industry and academia!!!

#### Start with Name: Title: Contribution. If the person contributed nothing then just put in writing: Did not participate in the project.
Your final model (model 3) and final results summary (this should be the last paragraph in D)
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

#### Your GitHub must be made public by the morning of the next day of the submission deadline.
Note: The 5 left over points will be awarded for participating in the voting event. Voting will be released and you will have 2 days to decide on your top 3 favorite projects.

