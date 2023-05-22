# Data Science

Data science is a field that involves analyzing and interpreting data to gain insights and make informed decisions. 
It uses techniques from statistics, mathematics, and computer science to extract meaningful information from data sets. 

Data scientists collect, clean, analyze, and visualize data to uncover non-obvious and useful patterns and trends from large datasets. 
The ultimate goal of data science is to use data to drive understanding, solve problems, and make data-driven decisions.

"Data Science is the art of turning data into actions."

![img.png](img/ds.png)

![img.png](img/ds1.png)

## Data
### Quantitive (numeric)
### Qualitative (categorical)
If I can't add something or can't get a meaningful result by adding or substracting the values of two variables, I am working with a categorical variable.
* I can't add cat + dog, female + male, married + single
* grades A, B and C can't be added until they are converted to numbers

Also, a categorical variable should divide the data into distinct groups or categories. 
Categorical variables typically have a limited number of unique values that define different groups or classes within the dataset.
* License plate numbers, although they may consist of letters and numbers, are not considered categorical variables because they do not represent distinct categories or groups. License plate numbers are typically used as unique identifiers rather than representing different groups or classes. 
* Emails. s. In most cases, analyzing or manipulating email addresses would involve tasks such as data validation, parsing, or extracting specific components (e.g., domain name, username) rather than treating them as categorical variables.
* ID numbers

Examples:
* hair color ("blonde", "brunette", "grey")
* cloud cover ("cloudy", "sunny", "partly cloudy")
* belief in life after death ("Yes", "No")
* In a football game ("Win", "Lose")
* Favourite artist ("Taylor Swift", "Coldplay")
* !! A variable using numbers as category labels is always a categorical variable

## Tidy Data
Tidy datasets provide a standardized way to link the structure of a dataset (its physical layout) with its semantics (its meaning).
* **Structure** is the form and shape of your data. In statistics, most datasets are rectangular data tables(data frames) and are made up of rows and columns.

* **Semantics** refers to the meaning of the data. Datasets consist of values, which can be either quantitative (numeric) or qualitative (categorical). These values are organized in two ways:

    * **Variables** — These are values that measure the same attribute across different units. For example, if you have a dataset of students, variables could be their age, test scores, or gender.
    * **Observations** — all values measured on the same unit across attributes. Continuing with the student example, observations would be individual students, and their attributes would be age, test scores, or gender.


![img.png](img/ds2.png)

There are three interrelated rules which make a dataset tidy:
1. Each variable must have its own column. 
2. Each observation must have its own row. 
3. Each value must have its own cell.

![img.png](img/td.png)