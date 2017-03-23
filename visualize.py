#For visualizing data frames

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def make_corr_matrix(df, file_name):
    ''' This function makes a correlation matrix with shades of red.  Floats and integers are formatted
        to fit in the matrix.
        df: data frame
        file_name:  file name to be saved
    '''
    correlation = df.corr().round(2)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(correlation, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    setting = "g"
    plt.figure()
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlation, mask=mask, annot=True,cmap='Reds', fmt = setting, linewidths=.5, ax=ax)
    plt.title("Correlations Between Variables", fontsize=20)
    plt.savefig(file_name)

