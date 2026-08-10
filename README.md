# Machine Learning from Scratch
This project folder contains SVM, K-Means, Mean-Shift and Neural Networks Models made from Scratch using Numpy and Python, with a small portion using Pandas and Scikit-Learn's `sklearn.preprocessing` for Data Cleaning. The purpose of this project folder was to delve deeper into the foundation of Supervised and Unsupervised Learning concepts as well as understand the mathematical and algorithmic complexity of the solutions.

Every algorithm is written out by hand with NumPy - no Scikit-Learn, PyTorch or TensorFlow for the model itself. Each notebook states what the algorithm does, walks through the class line by line, runs it on a real dataset and plots the result.

## Models
### K-Means Clustering
<hr>
`k-means_from_scratch.ipynb` - clusters the Iris flowers into 3 groups. The `K_Means` class keeps a dictionary of centroids and a dictionary of the points belonging to each one, then repeats two steps until nothing moves: put every point with its closest centroid, and move each centroid onto the average of its own points. Running it on Iris settles in 12 passes and lines up with the true species 88.7% of the time, with the elbow in the inertia curve correctly pointing at `k = 3`. Every mistake falls between `versicolor` and `virginica`, the two species that genuinely overlap.

### Support Vector Machine
<hr>
`support_vector_machine_from_scratch.ipynb` - separates `Iris-setosa` from `Iris-versicolor` using the two petal measurements. The `Support_Vector_Machine` class searches for the widest margin directly: it shrinks `w` step by step, keeps every candidate that satisfies `y * (w . x + b) >= 1` for all points, and takes the one with the smallest magnitude. It separates all 100 flowers correctly with a margin of 1.27. The notebook draws the boundary, the two margin lines and the support vectors that hold them in place. This search needs a problem a straight line can split perfectly, which is why those two clearly separated species were used.

### Mean Shift Clustering
<hr>
`mean_shift_from_scratch.ipynb` - separates skin from non-skin pixels without being told how many groups to look for. Every point starts as its own centroid and each pass moves it to a distance-weighted average of the points around it, so nearby points pull harder than distant ones, which walks the centroid to the nearest peak in the density. Centroids that land on the same peak are merged, and the number of survivors is the number of clusters. It is given a `bandwidth`, not a `k`.

This dataset was chosen because it is a case where K-Means fails. The skin and non-skin pixels form two long parallel bands in colour space, and the non-skin band is far wider and splits into several concentrations of its own. K-Means given the correct `k = 2` scores 79.9%, which is exactly the score for labelling every pixel non-skin - it cuts straight across both bands rather than along them, because splitting the large diffuse region lowers the total squared distance more than isolating the small tight one. Mean Shift at `bandwidth = 0.45` finds 8 clusters and reaches 96.6%, holding the skin pixels in two of them and leaving the rest pure. K-Means only catches up once it is given a `k` well above the number of classes, which means already knowing how the answer should come out.

### K-Nearest Neighbors
<hr>
`k_nearest_neighbors_from_scratch.ipynb` - classifies tumour samples as benign or malignant. There is no training step at all: the stored data is the model, and a prediction is just measuring the distance to every training point, keeping the `k` closest, and taking a majority vote. It reaches 97.8% at `k = 5` and 98.5% at `k = 7`, with `k = 1` the weakest setting at 94.9% because it copies its answer from a single neighbour with nothing to outvote it. The notebook draws one prediction in full, showing the 5 neighbours splitting 3-2 on a sample near the boundary, and maps the decision boundary, which follows the shape of the data rather than any fitted line.

### 2D Convolutional Neural Network
<hr>
`2d_cnn_from_scratch.ipynb` - classifies handwritten digits from 8x8 images. Every other model in this folder treats a row as a flat list of numbers, which throws away which pixels sit next to which. A convolution instead slides a small 3x3 filter across the image and records how strongly it matches at each position, so a stroke is recognised wherever it appears and the filter costs 9 weights no matter how large the image gets. The network is a convolution with 16 filters, ReLU, 2x2 max pooling, a flatten and a dense layer into 10 outputs with softmax - 1,610 weights in total.

Both directions are written out by hand. The forward pass loops over the 36 output positions and handles the whole batch at each one, and the backward pass reverses it: the gradient for a filter is the patch it was looking at, and pooling passes gradient back only to the value that won its block. A finite-difference check in the notebook confirms the hand-written derivatives match the loss to eight decimal places. It reaches 97.9% on the held-out digits after 30 epochs, which takes a couple of seconds. The learned filters are drawn out at the end - several came out as edge detectors without being told to.

### Linear Regression
<hr>
`linear_regression_from_scratch.ipynb` - predicts fuel economy in miles per gallon. The `Linear_Regression` class fits `y = w.x + b` by gradient descent on the mean squared error, and the notebook works through the R-squared theory: `SS_res`, `SS_tot`, why R-squared is the fraction of variance the model accounts for, why it can go negative on held-out data, and why adjusted R-squared exists. Weight alone gives a test R-squared of 0.766, and all seven features together give 0.843 at an RMSE of 3.19 mpg.

The closed-form normal equation is implemented alongside it as a check, and gradient descent lands on the same coefficients to within 5.4e-14. Feature scaling is shown rather than asserted: with raw features gradient descent either diverges to infinity in 28 iterations or, at a learning rate small enough to survive, crawls to an R-squared of -0.30. The noise-column experiment confirms that training R-squared rises every time a useless random feature is added while adjusted R-squared falls.

### Logistic Regression
<hr>
`logistic_regression_from_scratch.ipynb` - covers both the two-class and the many-class case. The `Logistic_Regression` class puts a sigmoid on a linear score, so the model is linear in the log-odds, and trains on binary cross-entropy. The `Softmax_Regression` class generalises it to three classes with the softmax, trained on categorical cross-entropy. The notebook derives why squared error is not used here, and why the sigmoid derivative cancelling against the log-loss derivative leaves the same clean gradient in both cases.

On the banknote data a single feature reaches 84.6% and all four reach 98.3%; the softmax reaches 98.2% on the Iris training set. The sigmoid and softmax functions are plotted in their own right, and the notebook asserts the identities directly - softmax rows sum to 1 to within 2.2e-16, and at two classes the softmax equals the sigmoid of the score difference with a maximum difference of exactly zero.

### Random Forest
<hr>
`random_forest_from_scratch.ipynb` - predicts the presence of heart disease. A `Decision_Tree` splits on `feature <= threshold`, picking the split that most reduces Gini impurity, and `Random_Forest` grows 100 of them, each on a bootstrap sample and each considering only a random subset of features at every split. Predictions are a majority vote. Trees split on thresholds, so no scaling is needed anywhere.

A single tree scores 1.000 on the training rows and 0.773 on the test set - it memorises. The forest reaches 0.867, and the out-of-bag score, which costs nothing extra because each tree skips about a third of the rows, comes to 0.833. The measured out-of-bag fraction is 0.370 against the predicted `(1 - 1/n)^n = 0.367`. The depth sweep shows the single tree peaking at depth 3 and decaying afterwards while the forest holds steady at every depth.

### XGBoost
<hr>
`xgboost_from_scratch.ipynb` - the same heart disease problem, so bagging and boosting can be compared directly. Where the forest grows independent trees in parallel to cut variance, boosting grows them in sequence, each fitted to the gradients left over by the ones before it. The notebook derives the second-order method properly: the Taylor expansion of the loss, the leaf weight `-G/(H+lambda)` that falls out of it, the similarity score, and the split gain with its `gamma` penalty.

Test log loss bottoms at 0.364 on round 65 and then climbs for the remaining rounds while training loss keeps falling - the overfitting signature that a random forest does not have. Of 277,616 candidate splits scored, 35.6% are rejected by `gamma` and only 932 are taken; 157 nodes had a genuinely positive gain and were still left as leaves. Splitting stops entirely at round 160, after which `gamma` prunes every tree down to a single leaf.

### Coming next
<hr>
Principal Component Analysis (PCA), Individual Component Analysis (ICA), t-SNE, GANs, Basic RecSys.
Reinforcement Learning

## Datasets 
### Iris Flower Classification 
<hr>
The <a href="https://www.neuraldesigner.com/learning/tutorials/data-set/#DataSource"> data </a> consists of 150 rows of Iris flower samples with 50 instances for each category - `Iris setosa`, `Iris versicolor`, and `Iris virginica`. It consists of 5 variables aka 5 columns with `petal-length`, `petal-width`, `sepal-length` and `sepal-width` as features and `class` as the output label. 

### Optical Recognition of Handwritten Digits
<hr>
The <a href="https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits"> data </a> contains 5,620 handwritten digits, roughly 560 of each digit from 0 to 9. Each row holds 64 values which are an 8x8 image flattened out, with each pixel running from 0 to 16, so reshaping a row back to 8x8 gives a picture that can be looked at directly. This is the only dataset in the folder where the features have a spatial layout rather than being an unordered list, which is what the convolutional network needs.

### Skin Segmentation
<hr>
The <a href="https://archive.ics.uci.edu/dataset/229/skin+segmentation"> data </a> contains 245,057 pixels sampled from face images, with `B`, `G` and `R` colour values as features and `class` as the output label where 1 is skin and 2 is not-skin. 50,859 of the pixels are skin and 194,198 are not. The clustering notebook works from a random sample of 800 rows, because comparing every point against every other point on each pass grows with the square of the row count. Only three features means distances stay meaningful, which is where density-based clustering works best.

### Breast Cancer Wisconsin
<hr>
The <a href="https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original"> data </a> contains 699 tumour samples with 9 features - `clump-thickness`, `uniformity-of-cell-size`, `uniformity-of-cell-shape`, `marginal-adhesion`, `single-epithelial-cell-size`, `bare-nuclei`, `bland-chromatin`, `normal-nucleoli` and `mitoses` - each recorded on a scale of 1 to 10, with `class` as the output label where 2 is benign and 4 is malignant. 16 rows have a missing `bare-nuclei` value and are dropped, leaving 683. Because every feature already shares the same scale, no scaling is needed before measuring distances.

### Auto MPG
<hr>
The <a href="https://archive.ics.uci.edu/dataset/9/auto+mpg"> data </a> contains 398 cars with 7 features - `displacement`, `cylinders`, `horsepower`, `weight`, `acceleration`, `model-year` and `origin` - and fuel economy in miles per gallon as the output label, ranging from 9.0 to 46.6. Six rows have a missing `horsepower` value and are dropped, leaving 392. This is the only dataset in the folder with a continuous target rather than a class label, which is what the regression notebook needs, and `weight` alone predicts it strongly enough to draw a fitted line through.

### Banknote Authentication
<hr>
The <a href="https://archive.ics.uci.edu/dataset/267/banknote+authentication"> data </a> contains 1,372 banknotes with 4 features taken from photographs - `variance`, `skewness`, `curtosis` and `entropy` of the wavelet-transformed image - and `class` as the output label, where the two classes are genuine and forged. The split is reasonably even at 762 to 610 and the classes are close to separable on the full feature set, which makes it a clean test for a linear classifier.

### Heart Disease Samples
<hr>
The <a href="https://archive.ics.uci.edu/dataset/45/heart+disease"> data </a> contains 303 instances of patient samples with 13 features determining the presence of a heart diease. Experiments have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0), however due to the usage on a clustering model, only 2 clusters for 0 (absence) and 1(presence) was used. Four rows are missing `ca` and two are missing `thal`, so 297 remain once they are dropped. The tree models use the same binary presence/absence target and need no scaling, since a tree splits on thresholds rather than distances. 

### Portuguese Bank Marketing Campaigns
<hr>
The <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing"> data </a> contains 45,211 records from direct marketing phone campaigns run by a Portuguese banking institution, with 16 features covering client details, contact details and prior campaign outcomes, and `y` as the output label recording whether the client subscribed to a term deposit. Only 11.7% of the records are positive, so it is kept for the models where that imbalance is worth working through.
