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
4. The result of df.isnull().sum() shows some features contain a bunch of null balue, we are concidering how to deal with this data, we will either completly remove them or refill them with generated value by some functions, we plan to do some search for that.
5. For the features contain value type string, we applied [unique()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=CpA9bV6xP9K9&line=1&uniqifier=1) function to make it easy to ditinguish.
6. For the binary feature like "sex", we devided them into intager 0 and 1 using function :
[df['sex'].replace('female', 0, inplace=True)](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=1&uniqifier=1)
8. For nonbinary features, we applied [OneHotEncoder()](https://colab.research.google.com/github/sebastian-dv/CSE-151A-Project/blob/main/SUPPORT2_Notebook.ipynb#scrollTo=89lDyDJ4QAdB&line=4&uniqifier=1) function
9. After above data exploration and preprocessing, we are able to apply some visualization tool to help us explore the pattern of data

## Visualization Tool
1. Parallel Coordinates Plot: we applied this function to visualize the relationship between "dementia" and other features.
2. Plotly Express interface: This function is used to observe the relationship between "age" and dimensions=['age', 'sps', 'meanbp','scoma','aps'], it seems like people's pysical feature changes around 50.
3. Multiple Line Plots: this function is used to check the pattern of two particular features.
   - first, we applied age-diabetes pair, it shows people between 40 to 80 is the main group to have diabetes
   - second, we applied bili-hrt pair and bili-ph pair, their diagram has similar pattern, and We think we should implement more data to see the pattern between them.
