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
After testing with the first model, the accuracy was low, at ~0.56, so we decided to try and improve our accuracy through other methods like K-fold and hypterparameter tuning.
Our attempt at using K-fold was not much better, at ~0.53 accuracy, but we still had a few more things to try.
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

Similar, to the inital version of the second model, this graph shows signs of overfitting, as seen by the disparity in loss apparent throughout the entire graph, which is different from the first model, as the first model was underfitting.
### Hyperparameter Tuned Model with Oversampling
![image](https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/3a5a0026-0896-4357-b8fb-7e49b2f5521d)
<img width="425" alt="image" src="https://github.com/sebastian-dv/CSE-151A-Project/assets/122483969/293e2b68-8e5f-4c04-a968-e003df5293c8">

This graph shows no major signs of underfitting or overfitting, as the losses are closely tied to each other, and progress at a similar rate, which is far better than the underfitting graph produced by the first model.

## Conclusion
Our second model has proved to be more accurate in predicting our target classes than our previous (multiclass) logistic regression model. Although, it started off very inaccurate and in some cases worse, we were able to apply different techniques such as K-fold, hyperparameter tuning, and oversampling, in order to improve the accuracy of our model. We believe that the largest contributor to this second model improving over the first was likely because of the oversampling we did. The oversampling helped balance out our target classes much more and helped our model learn much more effeciently. There is likely still room for improvement through more complex hyperparameter tuning and more time, but the improvement would probably only be very minimal over what we've achieved already. For our next model we are thinking of doing an SVM. We decided on an SVM because of its ability to support both categorical and numerical features, as well as it's ability to do multiclass classification. We want to possilby try a couple different kernel's in order to find which works best for our data.

## Jupyter Notebook Link
[Link to Jupyter Notebook in Github](https://github.com/sebastian-dv/CSE-151A-Project/blob/main/Milestone%204%3A%20Model%202.ipynb)
[Link to Jupyter Notebook in Google Colab](https://colab.research.google.com/drive/1quxLTGDJ_VTTTDzPrHU4nsm8mzdlV4yg?usp=sharing)
