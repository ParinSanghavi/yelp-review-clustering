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

* Clone the repository to some location in your machine
```
$ git clone git@github.ncsu.edu:arrao/yelp-review-parameter-extraction.git
$ cd yelp-review-parameter-extraction
```
* Copy the Dataset folder from the extracted dataset.zip mentioned in the previous section in this location
* Perform some basic basic pre-processing like segregating reviews into businesses. In order to segregate, enter y and hit return
```
$ cd Scripts
$ python cluster_yelp.py build
```
* This will create a folder in the parent directory called BusinessReview containing lots of files where reviews are segregated by businesses and businesses are segregated by city,state and category. It is not advisable to open this directory in a graphical file explorer as it might take a lot of time to list all the files.

##Execution Instructions
* All the scripts are in the Scripts directory within the project source code.
```
$  cd Scripts
```
* To view different operations that are possible run the following:
```
$ python cluster_yelp.py
```
* To view the list of cities, businesses, categories and states present in the dataset:
```
$ python cluster_yelp.py list city
$ python cluster_yelp.py list business
$ python cluster_yelp.py list category
$ python cluster_yelp.py list state
```
* To view the name of models already trained:
```
$ python cluster_yelp.py list models
```
* To view the parameter that can be passed to the list command:
```
$ python cluster_yelp.py list
```
* Prepare the dataset. If you have not segregated reviews by businesses press y else press n.
```
$ python cluster_yelp.py build
```
* Perform K-Means Clustering i.e. train the dataset. Once a model is trained, a folder with the models names is created in the Models directory in the parent directory. If these directories do not exist prior to execution, these directories will be created. train step takes 2 mandatory arguments and 1 optional argument. First argument is the path to the files containing reviews/businesses separated by commas. Multiple files result in intersection between the businesses. Second argument is number of clusters and the third optional argument is to perform latent semantic analysis or not before clustering. Default value is False. Some examples are given below. 
```
$ python cluster_yelp.py train 'path1,path2,..' <num_of_clusters> <lsa=False>
$ python cluster_yelp.py train ../BusinessReview/o9C4hmk1QfYPdgE0PytGkw.json 100 #single business
$ python cluster_yelp.py train '../BusinessReview/category_doctors.txt' 100 # all doctors category businesses
$ python cluster_yelp.py train '../BusinessReview/category_zoos.txt,../BusinessReview/city_phoenix.txt' 100 True # All zoos in Phoenix with lsa True
```
* Analyze the clusters formed from training. This takes one mandatory argument - model name and one optional argument segregate which defaults to True. model name is the name of the model to be analyzed. The list of models can be obtained by using the list models command as described above. The segregate option segregates the sentence fragments in the dataset into appropriate clusters (Cluster_<ID>.txt files) creating a folder Clusters within the Models/Model-Name directory. This should be performed at least once. Labelling, Merging and Ranking of clusters happens in this step.
```
$ python cluster_yelp.py analyze <model-name> <segregate=True>
$ python cluster_yelp.py analyze o9C4hmk1QfYPdgE0PytGkw.json #single business
$ python cluster_yelp.py analyze 'c#phoenix@ca#zoos' #zoos in phoenix 
```
* View the results for a trained model. This results in hiererchical display of clusters along with levelwise clusters ranked in non-increasing order of score. It takes only one argument namely model-name. 
```
$ python cluster_yelp.py view <model-name>
$ python cluster_yelp.py view o9C4hmk1QfYPdgE0PytGkw.json #single business
$ python cluster_yelp.py view 'c#phoenix@ca#zoos' #zoos in phoenix 
```
* Perform all operations in a single command. It takes the exact arguments as the train command described above.
```
$ python cluster_yelp.py complete 'path1,path2,..' <num_of_clusters> <lsa=False>
$ python cluster_yelp.py complete ../BusinessReview/o9C4hmk1QfYPdgE0PytGkw.json 100 #single business
$ python cluster_yelp.py complete '../BusinessReview/category_doctors.txt' 100 # all doctors category businesses
$ python cluster_yelp.py complete '../BusinessReview/category_zoos.txt,../BusinessReview/city_phoenix.txt' 100 True # All zoos in Phoenix with lsa True
```
* Decoding Model Names: A model for single business is named by just the business id. On the other hand for criteria such as a category in a specific city/state some rules are used for generating the model name:
  1. the naming occurs in the order state, city and category. 
  2. state is prefixed by s, city is prefixed by c and category is prefixed by ca
  3. All the names of a particular type are separated by #
  4. All different types of criteria are seprated by @
```
s#nv@ca#restaurants : State: NV and Category : Restaurants
c#phoenix@ca#zoos : City : Phoenix Category : Zoos
o9C4hmk1QfYPdgE0PytGkw : Business ID : o9C4hmk1QfYPdgE0PytGkw
ca#weight_loss_centers : Category : weight_loss_centers
```
