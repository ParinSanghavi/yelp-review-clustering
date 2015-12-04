# Yelp Review Clustering Tool v0.1

## Mining Quality Parameters from Yelp Review To Improve Businesses

##Authors:

1. Akhil Raghavendra Rao (arrao@ncsu.edu)
2. Parin Rajesh Sanghavi (prsangha@ncsu.edu)
3. Dharmendrasinh Jaswantsinh Vaghela (djvaghel@ncsu.edu)

##Libraries:

1. Numpy : Installed with scipy
2. Scipy : http://www.scipy.org/install.html
3. Scikit-Learn : http://scikit-learn.org/stable/install.html
4. Pandas : http://pandas.pydata.org/pandas-docs/stable/install.html
5. Matplotlib : http://matplotlib.org/users/installing.html
6. NLTK : http://www.nltk.org/install.html

##Getting the dataset:

1. The dataset can be downloaded from here: https://drive.google.com/a/ncsu.edu/file/d/0B88KCEO9WlRUT3B1aTZ3azVDLXM/view?usp=sharing
2. Unzip the dataset. Place the Dataset folder in the same location as the Scripts folder which is part of this repository
3. The directory should look something like this...
```
<location>
---Scripts/
------cluster_yelp.py
------yelp.py
------divide_by_business.py
---Dataset/
------business.json
------review.json
```

##Setting up the environment:

1. Clone the repository to some location in your machine
```
$ git clone git@github.ncsu.edu:arrao/yelp-review-parameter-extraction.git
$ cd yelp-review-parameter-extraction
```
2. Copy the Dataset folder from the extracted dataset.zip mentioned in the previous section in this location
3. Perform some basic basic pre-processing like segregating reviews into businesses. In order to segregate, enter y and hit return
```
$ cd Scripts
$ python cluster_yelp.py build
```
4. This will create a folder in the parent directory called BusinessReview containing lots of files where reviews are segregated by businesses and businesses are segregated by city,state and category. It is not advisable to open this directory in a graphical file explorer as it might take a lot of time to list all the files.

##Execution Instructionsy
1. All the scripts are in the Scripts directory within the project source code.
```
$  cd Scripts
```
2. To view different operations that are possible run the following:
```
$ python cluster_yelp.py
```
3. To view the list of cities, businesses, categories and states present in the dataset:
```
$ python cluster_yelp.py list city
$ python cluster_yelp.py list business
$ python cluster_yelp.py list category
$ python cluster_yelp.py list state
```
4. To view the name of models already trained:
```
$ python cluster_yelp.py list models
```
5. To view the parameter that can be passed to the list command:
```
$ python cluster_yelp.py list
```
6. Prepare the dataset. If you have not segregated reviews by businesses press y else press n.
```
$ python cluster_yelp.py build
```
7. 
