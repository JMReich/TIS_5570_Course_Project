# This is how you add a comment in Python

'''
This is also a comment in Python, but in this case it is a multi-line comment

Just don't forget to close it with three single quotes.

'''




'''
General Information:


At a minimum, you will need to answer three of the following business questions:
1. What is the expected selling price of my home?
2. What factors influence the price of my home?
3. Which factors are more important than others?
4. How much should I invest in improving the condition of my home in order to
increase the expected price by more than the cost of improvements?
5. Which homes should I compare my house to?
6. When is the best time of the year to sell my home?

You are encouraged to create additional questions and answer them. Your goal is to
bring insights and questions that the AREA team bas not thought about.



Dataset and Supporting Files
The Ames dataset that you must use in this project is in a Canvas module called
"Course Project Files". In this module, you will find the following files:
Ames.xlsx: This is the ames data set
- Ames Data Dictionary-final.pdf: This is a file containing a description of the
data elements/columns contained in the ames.xlsx file
- Modeling Home Prices Using Realtor Data.pdf: This article provides insights
into how to model home prices.
- Modeling references.pdf: This file contains links to references about how to
analyze the ames data.
- You can also do your research and find other material related to the ames
dataset. Please reference material that is not your work and include a
citation in your report.

'''


# Importing the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Allowing all columns and rows to be displayed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Reading the data from the excel file
df = pd.read_excel('./Data/ames.xlsx')

# Printing the count of rows for each column


def missing_data_point_count(df):
    total_rows = 2930
    data_points = []
    processed_columns = set()

    for column in df.columns:
        if column not in processed_columns:
            count = df[column].isna().sum()
            percentage = (count / total_rows) * 100
            data_points.append((column, count, percentage))
            processed_columns.add(column)

    # Sort by the number of data points in descending order
    data_points.sort(key=lambda x: x[1], reverse=True)

    # Print the output in three columns
    print(f"{'Count':<3} {'Column Name':<15} {'Data Points':<11} {'Percentage':<10}")
    print("="*55)
    id = 1
    for column, count, percentage in data_points:
        print(f" {id:<4} {column:<15} {count:<12} {percentage:<10.4f}")
        id += 1
        
def mean_numeric_columns(df):
    # Calculate the mean of each numeric column and print
    numeric_cols = df.select_dtypes(include=['number']).columns
    mean_values = df[numeric_cols].mean()
    
    print(f"{'Column Name':<15} {'Mean':<10}")
    print("="*25)
    for column, mean in mean_values.items():
        print(f"{column:<15} {mean:<10.4f}")

def skewness(df):
    # Calculate the skewness of each numeric column and print
    numeric_cols = df.select_dtypes(include=['number']).columns
    skewness_values = df[numeric_cols].skew()
    
    print(f"{'Column Name':<15} {'Skewness':<10}")
    print("="*25)
    for column, skew in skewness_values.items():
        print(f"{column:<15} {skew:<10.4f}") 
        
def plot_skewness(df, column):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        print(f"Column '{column}' does not exist in the DataFrame.")
        return
    
    # Drop NA values for the column
    data = df[column].dropna()
    
    # Calculate skewness
    skewness_value = skew(data)
    
    # Plot the histogram
    plt.hist(data, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'{column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
        
def plot_misc_feature_vs_sale_price(df):
    # Replace NA values in Misc_Feature with the word "missing"
    df['Misc_Feature'] = df['Misc_Feature'].fillna('missing')
    
    # Group by Misc_Feature and calculate the mean SalePrice
    mean_sale_price = df.groupby('Misc_Feature')['SalePrice'].mean()

    # Plot the results
    mean_sale_price.plot(kind='bar', color='skyblue')
    plt.xlabel('Misc Feature')
    plt.ylabel('Mean Sale Price')
    plt.title('Mean Sale Price by Misc Feature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_correlation_heatmap(df):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=['number'])
    
    # Calculate the correlation matrix
    corr_matrix = numeric_cols.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numeric Columns')
    plt.show()

def plot_sale_price_distribution(df):
    # Check if the SalePrice column exists in the DataFrame
    if 'SalePrice' not in df.columns:
        print("Column 'SalePrice' does not exist in the DataFrame.")
        return
    
    # Drop NA values for the SalePrice column
    data = df['SalePrice'].dropna()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='skyblue')
    plt.title('Distribution of SalePrice')
    plt.xlabel('SalePrice')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
def plot_sale_price_boxplot(df):
    # Check if the SalePrice column exists in the DataFrame
    if 'SalePrice' not in df.columns:
        print("Column 'SalePrice' does not exist in the DataFrame.")
        return

    # Drop NA values for the SalePrice column
    data = df['SalePrice'].dropna()

    # Plot the box and whisker plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=data, color='skyblue')
    plt.title('Box and Whisker Plot of SalePrice')
    plt.xlabel('SalePrice')
    plt.grid(True)
    plt.show()

def plot_sale_price_scatter(df):
    # Check if the SalePrice and Gr_Liv_Area columns exist in the DataFrame
    if 'SalePrice' not in df.columns or 'Gr_Liv_Area' not in df.columns:
        print(f"Column 'SalePrice' or 'Gr_Liv_Area' does not exist in the DataFrame.")
        return

    # Drop NA values for the SalePrice and Gr_Liv_Area columns
    data = df[['SalePrice', 'Gr_Liv_Area']].dropna()

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data['Gr_Liv_Area'], y=data['SalePrice'], color='skyblue')
    plt.title('Scatter Plot of SalePrice vs Ground Living Area')
    plt.xlabel('Ground Living Area (Gr_Liv_Area)')
    plt.ylabel('SalePrice')
    plt.grid(True)
    plt.show()
    
def plot_neighborhood_counts(df):
    # Check if the Neighborhood column exists in the DataFrame
    if 'Neighborhood' not in df.columns:
        print("Column 'Neighborhood' does not exist in the DataFrame.")
        return
    
    # Count the number of houses in each neighborhood
    neighborhood_counts = df['Neighborhood'].value_counts().sort_values(ascending=True)
    
    # Generate a larger set of colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(neighborhood_counts)))
    
    # Plot the horizontal bar chart
    plt.figure(figsize=(12, 8))
    neighborhood_counts.plot(kind='barh', color=colors)
    plt.xlabel('Number of Houses')
    plt.ylabel('Neighborhood')
    plt.title('Number of Houses in Each Neighborhood')
    plt.tight_layout()
    plt.show()
    

# LINEAR REGRESSION JMR



# Assuming df is your DataFrame and 'SalePrice' is the target variable
# Replace 'Full_Bath', 'Fireplaces', 'Garage_Type', 'Kitchen_Qual' with the actual feature column names
X = df[['Full_Bath', 'Fireplaces', 'Garage_Type', 'Kitchen_Qual']]  # Features
y = df['SalePrice']  # Target variable

# Convert categorical variables to numerical values using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=100)

# Create the linear regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')





# This is how you print the first 5 rows of a dataframe and last 5 rows
# print(df)

# # JMR - This is how you access a specific cell in a dataframe by index
# print(df.iloc[0, 2]) 
# # JMR - This is more specific and but you need to remember all of the column names or make an array for them
# print(df.iloc[1, df.columns.get_loc('Lot_Shape')]) 

# df_lots = df[['Lot_Shape', 'Lot_Config', 'Lot_Area', 'Lot_Frontage']]
# print(df_lots.count())
# grouped_lots = df_lots.groupby(['Lot_Shape'])
# # grouped_lots.mean()

# print(grouped_lots)






















