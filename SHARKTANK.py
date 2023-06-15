#!/usr/bin/env python
# coding: utf-8

# # IMPORTING DATASET

# In[35]:


import pandas as pd #pandas work with relational data
data=pd.read_csv(r"C:\\Users\D C ARYA\Downloads\SharkTank.csv")


# In[36]:


data


# In[ ]:





# # BASIC FUNCTIONS

# In[37]:


data.head() #to show top 5 records of dataset


# In[38]:


data.tail() #to show bottom 5 records of dataset


# In[39]:


data.shape  #to show no. of rows and columns


# In[40]:


data.size  #to show no. of total elements in the dataset


# In[41]:


data.columns #to show each column name


# In[42]:


data.dtypes #to show data-type of each column


# In[43]:


data.info()  #to show indexes, columns, data-types of each columns, memory at once


# In[ ]:





# # DATA CLEANING

# In[44]:


data.shape


# In[45]:


data[data.duplicated()] #to check row wise and detect duplicate rows


# In[46]:


data.drop_duplicates() #remove the duplicate rows permanently


# In[47]:


data.drop_duplicates(inplace=True)


# In[48]:


data[data.duplicated()]


# In[49]:


data.shape


# In[50]:


data.to_csv('Shark1.csv', index=False) #modified csv


# In[51]:


data


# In[ ]:





# # DATA VISUALIZATION

# In[52]:


data.groupby('Industry').Industry.count() #group all unique items of column and show their count


# In[53]:


import pandas as pd  
import seaborn as sns #for creative plots
import matplotlib.pyplot as plt #data visulization library for 2d plots of arrays

column = data['Episode Number']

# Create the histogram
plt.hist(column, bins=10, edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Episode Number')
plt.ylabel('Frequency')
plt.title('Histogram of Episode Number')
plt.show()


# In[54]:


deal_counts = data['Industry'].value_counts()

# Display the counts
print(deal_counts)


# In[55]:


industry_counts = data['Industry'].value_counts()

# Extract the industry names and counts
industries = industry_counts.index
counts = industry_counts.values

# Create the bar plot
plt.bar(industries, counts)

# Set the axis labels and title
plt.xlabel('Industry')
plt.ylabel('Count')
plt.title('Count of Each Industry')
plt.xticks(rotation=90)

# Display the bar graph
plt.show()


# In[56]:


avg_sharks_by_industry = data.groupby('Industry')['Number of sharks in deal'].mean().reset_index()
avg_sharks_by_industry = avg_sharks_by_industry.sort_values('Number of sharks in deal', ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(avg_sharks_by_industry['Industry'], avg_sharks_by_industry['Number of sharks in deal'])
plt.xlabel('Industry')
plt.ylabel('Average Number of Sharks')
plt.title('Average Number of Sharks in a Deal by Industry')
plt.xticks(rotation=90)
plt.show()


# In[57]:


industry_counts = data['Industry'].value_counts()
accepted_offer_counts = data[data['Accepted Offer'] == 1]['Industry'].value_counts()

# Calculate the proportions
proportions = accepted_offer_counts / industry_counts

# Sort the proportions in descending order
proportions = proportions.sort_values(ascending=False)

# Create the bar plot
plt.bar(proportions.index, proportions)
plt.xlabel('Industry')
plt.ylabel('Proportion of Accepted Offers')
plt.title('Proportion of Accepted Offers by Industry')

# Rotate the x-axis labels if needed
plt.xticks(rotation=90)

# Display the bar graph
plt.show()


# In[58]:


# Assuming the column name is 'Pitchers city'
pitchers_city_column = 'Pitchers City'

# Split multiple cities and create a new DataFrame with individual cities
city_data = data[pitchers_city_column].str.split(',', expand=True).stack().reset_index(level=1, drop=True).to_frame(name='City')

# Count the occurrences of each city
city_counts = city_data['City'].value_counts()

# Filter out non-Indian cities if needed
indian_cities = city_counts[city_counts.index.str.contains('Delhi|Ahmedabad|Bangalore|Mumbai|Chennai|Kolkata|Hyderabad|Pune', case=False)]

# Plot the pie chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, _, _ = ax.pie(indian_cities, startangle=90, autopct='%1.1f%%', pctdistance=0.85)
# Create a legend
ax.legend(wedges, indian_cities.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)

plt.title('Proportion of Indian Cities')
plt.show()


# In[79]:


# Assuming the column name is 'Pitchers Average Age'
average_age_column = 'Pitchers Average Age'

# Count the occurrences of each category
category_counts = data[average_age_column].value_counts()

# Create a pie chart
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%')
plt.title('Proportion of Pitchers Average Age Categories')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[24]:


x = data['Yearly Revenue']
y = data['Monthly Sales']

# Create the scatterplot
plt.scatter(x, y, s=50)  # Increase the 's' value to make the dots larger
plt.xlabel('Yearly Revenue')
plt.ylabel('Monthly Sales')
plt.title('Scatterplot of Monthly Sales by Yearly Revenue')
plt.show()


# In[25]:


selected_columns = [ 'Number of sharks in deal']

# Creating the box plot with selected columns
data[selected_columns].plot(kind='box', figsize=(20, 10))


# In[ ]:





# In[26]:


import plotly.graph_objects as go
import plotly.io as pio


# In[ ]:





# # STATISTICAL ANALYSIS

# In[28]:


# Calculate the mean for each gender separately
mean_male = data['Male Presenters'].mean()
mean_female = data['Female Presenters'].mean()
mean_transgender = data['Transgender Presenters'].mean()

# Create a table to display the mean values
gender_means = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Transgender'],
    'Mean': [mean_male, mean_female, mean_transgender]
})

# Print the table
print(gender_means)





# In[29]:


median_male = data['Male Presenters'].median()
median_female = data['Female Presenters'].median()
median_transgender = data['Transgender Presenters'].median()

# Create a table to display the mean values
gender_median = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Transgender'],
    'Median': [median_male, median_female, median_transgender]
})

# Print the table
print(gender_median)


# In[30]:


# Perform value counts for each gender column
# Calculate the total counts of each gender
total_male_counts = data['Male Presenters'].value_counts().sum()
total_female_counts = data['Female Presenters'].value_counts().sum()
total_transgender_counts = data['Transgender Presenters'].value_counts().sum()

# Display the total counts
print("Total Male Presenters:", total_male_counts)
print("Total Female Presenters:", total_female_counts)
print("Total Transgender Presenters:", total_transgender_counts)




# In[31]:


import seaborn as sns


# In[32]:


heatmap_data = data[['Number of sharks in deal', 'Yearly Revenue', 'Total Deal Amount', 'Total Deal Equity']]


# In[33]:


correlation_matrix = heatmap_data.corr()


# In[93]:


# Select the columns for the correlation heatmap
selected_columns = ['Season Number', 'Episode Number', 'Pitch Number', 'Number of Presenters', 'Male Presenters', 'Female Presenters']

# Create a subset of the data with only the selected columns
subset_data = data[selected_columns]

# Create a correlation matrix
corr_matrix = subset_data.corr()

# Generate a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:





# In[ ]:





# # MACHINE LEARNING ALGORITHM

# # Decision Tree Classifier

# In[94]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



# In[95]:


# Select the relevant columns for the classifier
features = ['Season Number', 'Episode Number', 'Number of Presenters', 'Male Presenters', 'Female Presenters', 'Couple Presenters', 'Yearly Revenue', 'Monthly Sales']
target = 'Accepted Offer'

# Remove rows with missing values in the selected features and target
data = data.dropna(subset=features + [target])

# Split the data into features (X) and target variable (y)
X = data[features]
y = data[target]


# In[96]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[97]:


clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[98]:


y_pred = clf.predict(X_test)


# In[99]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Generate a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)


# In[ ]:





# # Regression

# In[34]:


#how dependent value is changing corresponding to an independent variables
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split #scikit-learn is used to split the data into training and testing sets.
from sklearn.metrics import accuracy_score, classification_report


# In[101]:


# Select the relevant columns for the logistic regression
features = ['Season Number', 'Episode Number', 'Number of Presenters', 'Male Presenters', 'Female Presenters', 'Couple Presenters', 'Yearly Revenue', 'Monthly Sales']
target = 'Accepted Offer'

# Remove rows with missing values in the selected features and target
data = data.dropna(subset=features + [target])

# Split the data into features (X) and target variable (y)
X = data[features]
y = data[target]


# In[102]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #parameter is set to 0.2, which means that 20% of the data will be used for testin
# the remaining 80% will be used for training.


# In[103]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[104]:


y_pred = logreg.predict(X_test)


# In[105]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Generate a classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

#The accuracy score of 0.33 indicates that the classifier achieved an accuracy of 33.33% on the test set. 
#This means that the classifier's predictions matched the actual 'Accepted Offer' values in the test set 
#for approximately one-third of the instances.


# In[ ]:


#The accuracy of the decision classifier is 66% and the accuracy of the regression model is 33%.

#In the context of machine learning, accuracy is a commonly used metric to evaluate the performance of a model. 
#It represents the proportion of correct predictions made by the model out of the total number of predictions.

#Decision Classifier Accuracy: 
#An accuracy of 66% for the decision classifier means that it correctly predicted the target variable (in this case, 'Accepted Offer') for 66% of the instances in the test set. 
#This suggests that the decision classifier performed better than random chance in predicting the target variable.

#Regression Model Accuracy: 
#An accuracy of 33% for the regression model means that it correctly predicted the target variable for 33% of the instances in the test set. 
#In the case of a regression model, accuracy is calculated by comparing the predicted continuous values to the actual continuous values of the target variable. 
#It indicates that the regression model had a lower predictive accuracy compared to the decision classifier.


# In[ ]:




