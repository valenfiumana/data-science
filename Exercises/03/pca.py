import pandas as pd     # for manipulating data
import numpy as np      # for generating random numbers
import random as rd     # for generating an example dataset (won't need it when using real data)
from sklearn.decomposition import PCA   # importing PCA function
from sklearn import preprocessing       # importing functions for scaling the data before performing PCA
from sklearn.preprocessing import StandardScaler # for scaling data
import matplotlib.pyplot as plt         # for fancy graphs


# 1. Generate a sample dataset
genes = ['gene' + str(i) for i in range (1, 101)]   # generate 100 gene names: "gene1", "gene2", ...
wt = ['wt' + str(i) for i in range (1, 6)] # create arrays of sample names: 5 "wt" == "wild type"
ko = ['ko' + str(i) for i in range (1, 6)] # 5 "ko" == "knock out"
data = pd.DataFrame(columns=[*wt, *ko], index=genes)    # The * unpack de "wt" and "ko" arrays so that the column names are a single array
                                                        # [wt1, wt2, wt3, wt4, wt5, ko1, ko2, ko3, ko4, ko5]
                                                        # Without the * it would create this array:
                                                        # [ [wt1, wt2, wt3, wt4, wt5], [ko1, ko2, ko3, ko4, ko5] ]
                                                        # the genes names are used as index == row names

for gene in data.index: # create random data
    data.loc[gene, 'wt1':'wt5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5) # for each gene in the index (gene1, gene2, ...) we create 5 "wt" and 5 "ko" values
    data.loc[gene, 'ko1':'ko5'] = np.random.poisson(lam=rd.randrange(10, 1000), size=5)

# print(data.head()) # first 5 rows of the data

#        wt1  wt2  wt3  wt4  wt5  ko1  ko2  ko3  ko4  ko5
# gene1  610  600  580  589  630  429  465  457  441  461
# gene2   23   29   24   35   24  561  526  543  583  580
# gene3   37   32   50   33   35  412  419  437  412  405
# gene4  691  685  769  740  796  889  875  900  859  916
# gene5  628  636  597  574  643  693  763  764  703  770

print(data.shape) # dimensions of data matrix: (100, 10) == 100 genes, 10 samples

# 2. Preprocessing
scaled_data = preprocessing.scale(data.T)   # Before doing PCA, we have to center and scale the data
                                            # After centering, the avg value for each gene will be 0
                                            # After scaling, the standard deviation for the values for each gene will be 1
                                            # We are passing the transpose of the data (data.T) bc the scale function expects samples to be rows
# scaled_data = StandardScaler().fit_transform(data.T) # This is another method to center and scale, whic is more commonly used

# 3. PCA
pca = PCA()             # create PCA object. sklearn uses objects that can be trained using one dataset and applied to another dataset
pca.fit(scaled_data)    # here we do all PCA math (calculate loading scores and the variation each principal component accounts for)
pca_data = pca.transform(scaled_data)   # generate coordinates for a PCA graph based on loading scores and scaled data

# 4. Scree plot
per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)  # The scree plot shows the percentage of explained variance for each principal component.
                                                                    # The code calculates the explained variance ratio and multiplies it by 100 to obtain percentages
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)] # create labels for the scree plot "PC1", "PC2", ...

plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

# 5. PCA Plot
# to draw a PCA plot, we'll first put the new coordinates created by pca.transform(scaled_data) into a matrix
# where the rows have sample labels and the columns have PC labels
pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)

plt.scatter(pca_df.PC1, pca_df.PC2) # Create a scatter plot using PC1 and PC2 values
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0])) # Set the label for the x-axis with the percentage of explained variance for PC1
plt.xlabel('PC2 - {0}%'.format(per_var[1]))

for sample in pca_df.index: # Annotate each sample in the scatter plot
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()

# The wt samples clustered on the left side, suggesting that they are correlated with each other
# The ko samples clustered on the right side, suggesting that they are correlated with each other
# And the separation between the two clusters along de x-axis suggests that wt and ko samples are very different from each other