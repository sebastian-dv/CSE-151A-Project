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

## Conclusion
### First Model
The first model is not the most precise, as there is a relatively clear sign of overfitting due to the cross-validation score being lower in the start than the training score, and only a relative evening out towards the end of the graph. We can possibly improve this model by selecting different features for training use to further finetune the results and not have overfitting or underfitting for the model.

### Visualizaiton 
![Screenshot 2024-03-10 at 7 16 17 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/90e3f1ea-bb2b-4f66-ad46-c7322d1a7d89)
This is our plot for our multi-class regression model, comparing our training score vs our cross validation score.
This plot shows that there is some overfitting in our model and thus a logistic regression model likely isn't the best model we could use for our data, which is usefult to know for our future models. We also had other similar plots for our logistic regression models that we did for each target rather than multiple at once.

![Screenshot 2024-03-10 at 7 20 21 PM](https://github.com/sebastian-dv/CSE-151A-Project/assets/23327980/f9e92ef8-55ad-4b95-ad17-ad58568ef99d)
Here is one of those models, the model itself doesn't look too different from the other models for each target and the results are relatively the same.

## MS2
## Requirements:
In this milestone you will focus on building your second model. You will also need to evaluate your this model and see where it fits in the underfitting/overfitting graph.

1.  Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.

2. Train your second model

3. Evaluate your model compare training vs test error

4. Where does your model fit in the fitting graph, how does it compare to your first model?

5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?

5. What is the plan for the next model you are thinking of and why?

6. Update your readme with this info added to the readme with links to the jupyter notebook!

7. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?

## Our Work
Our Second model is a Neural Network model.

First, we tried manually creating a 4-layer model(including input and output layers) with the 'tanh' activation function for the input and hidden layer, and the 'softmax' activation function for the output layer since we got encoded target y. We observed results by classification report and plot which simulate the distance between our testing result and training input data.

But it turns out the accuracy was low, so we decided to add an early_stopping, trying to improve the accuracy of our model.

Then we build a hyperparameter tuning model by tuning the units, activation function, and learning rates, trying to find the optimized parameters to rebuild our model, we sort our tuner result by the score of accuracy.

After finding the better model parameter, we rebuilt our model, printed the classification, and displayed the plot again, it turns out we improved our accuracy from 0.5 to 0.7, but we still looking for a better accuracy.

Then we decided to apply OverSamplying since we find our entries are imbalanced.

After applying the RandomOverSampler, we got 0.8ish accuracy.

## Visualization Tool
## Conclusion
