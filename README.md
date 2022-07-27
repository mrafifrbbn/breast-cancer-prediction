# Context

Hello! This is my second data science project. Here, I make machine learning models to predict whether a breast cancer is benign or malignant based on the available features. I try to implement what I have learned so far: standard data cleaning, EDA, classification model, hyperparameter tuning, and dimensionality reduction with PCA.

# Files

This repo contains this `README`, a python notebook called `breast_cancer.ipynb`, a slides in pdf format to highlight the most important aspects of this project called 'Breast Cancer Prediction.pdf', and the dataset `data.csv` is in the `data` directory.

# Data

I use the Breast Cancer Wisconsin dataset obtained from [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). This dataset is also one of the [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html) readily available in the scikit-learn package. The content of this dataset is as follows:

Attribute columns:
1) `id` - the patient's ID number
2) `diagnosis` - (M = malignant, B = benign)

Ten real-valued features directly computed from images: <br>
a) `radius` - mean of distances from center to points on the perimeter  <br>
b) `texture` - standard deviation of gray-scale values <br>
c) `perimeter` <br>
d) `area` <br>
e) `smoothness` - local variation in radius lengths <br>
f) `compactness` - computed as perimeter^2 / area - 1.0 <br>
g) `concavity` - severity of concave portions of the contour <br>
h) `concave points` - number of concave portions of the contour <br>
i) `symmetry` <br>
j) `fractal_dimension` - computed as "coastline approximation" - 1 <br>

In addition, each of these ten-valued features have three measurements: the mean value, standard error, and "worst" or largest (mean of the three largest values) of these features,resulting in 30 features. For instance, field 3 is Mean Radius (`radius_mean`), field 13 is Radius SE (`radius_se`), and field 23 is Worst Radius (`radius_worst`).

All feature values are recoded with four significant digits.

# Methods

I use point biserial correlation to pick the stronger predictor, i.e. `_worst`. Since the two classes are linearly separable, I use logistic regression and SVM as the classifier. To get the optimum results, I employ the grid search method to obtain the best hyperparameters for each model. Next, I use PCA to reduce the dimensionality of the data to see if the results can be improved.

To run the .ipynb file, the standard `numpy`, `pandas`, `matplotlib`, `seaborn`, and `sklearn` packages are required.

# Results

Both logistic regression and SVM perform extremely well on this dataset, with a recall score of 98% for logistic regression and a slightly lower score of 97% for SVM. There are no significant improvements on the recall scores if the dimensions are reduced with PCA, but the feature selection is made very efficient and easy.
