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

### Coming next
<hr>
3D CNN, Recurrent Neural Networks and Transformers.

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

### Heart Disease Samples
<hr>
The <a href="https://archive.ics.uci.edu/dataset/45/heart+disease"> data </a> contains 303 instances of patient samples with 13 features determining the presence of a heart diease. Experiments have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0), however due to the usage on a clustering model, only 2 clusters for 0 (absence) and 1(presence) was used. 

### Portuguese Bank Marketing Campaigns
<hr>
The <a href="https://archive.ics.uci.edu/dataset/222/bank+marketing"> data </a> contains 45,211 records from direct marketing phone campaigns run by a Portuguese banking institution, with 16 features covering client details, contact details and prior campaign outcomes, and `y` as the output label recording whether the client subscribed to a term deposit. Only 11.7% of the records are positive, so it is kept for the models where that imbalance is worth working through.
