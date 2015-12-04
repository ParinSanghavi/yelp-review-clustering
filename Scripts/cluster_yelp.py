import sys
import os
import pickle
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import scipy
from yelp import *
import nltk
from sklearn.decomposition import TruncatedSVD
from nltk.probability import FreqDist
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.cm as cm

######################################################
# Yelp Review Clustering Tool v0.1                   #
# Authors: Akhil Rao (arrao@ncsu.edu),               #
#          Parin Sanghavi (prsangha@ncsu.edu),       # 
#          Dharmendrasinh Vaghela (djvaghel@ncsu.edu)#                                  
######################################################


stemmer=nltk.PorterStemmer()

nounsfile=open('nouns.pickle','r')
nouns=pickle.load(nounsfile)
nounsfile.close()

cities=[]
categories=[]
businesses=[]
states=[]
trained=[]
try:
    fcity=open('cities.pickle','r')
    #print "Loading cities data"
    cities=pickle.load(fcity)
    fcity.close()
except:
    print "Cities data not generated"
try:
    fbusiness=open('businesses.pickle','r')
    #print "Loading Business data"
    businesses=pickle.load(fbusiness)
    fbusiness.close()
except:
    print "Businesses data not generated"
try:
    fstate=open('states.pickle','r')
    #print "Loading State wise data"
    states=pickle.load(fstate)
    fstate.close()
except:
    print "States data not generated"

try:
    fcategory=open('categories.pickle','r')
    #print 'Loading Categories data'
    categories=pickle.load(fcategory)
    fcategory.close()
except:
    print "Category data not loaded"

def refresh_trained():
    if not os.path.exists('../Models/'):
        os.mkdir('../Models')
    global trained
    trained = os.listdir('../Models')

refresh_trained()

class Cluster(object):
    
    def __init__(self):
        self.number=-1
        self.label=""
        self.score=0.0
        self.leaf=True
        self.children={}
        self.token_count=0.0
        self.label_frequency=0.0
        self.sentences_count=0.0
        self.total_sentences=0.0
        self.threshold=1.0
        self.level=0
        self.files=[]
        self.root=False

    def get_labels_levelwise(self):
        levels=[]
        for i in range (0,self.level):
            levels.append([])
        queue=[self]
        while len(queue)>0:
            current=queue[0]
            del queue[0]
            for child in current.children.values():
                if child.level >=0 and child.level <= self.level:
                    levels[child.level].append(child)
                    queue.append(child)    
        for level in levels:
            level.sort(key=lambda x: -x.score)
        return levels

    @classmethod
    def build(cls,labels,thresholds=[0.3,0.2,0.15]):
        thresholds.sort(reverse=True)
        d={}
        for label in labels:
            key=label[0]
            c=Cluster()
            d[key]=c
            c.label=label[1]
            c.score=label[-1]
            c.number=key
            c.leaf=True
            c.label_frequency=label[2]
            c.sentences_count=label[4]
            c.token_count=label[3]
            c.total_sentences=label[5]
            c.files.append('Cluster_%d.txt'%key)
        level=1
        merged=list(labels)
       
        for thr in thresholds:
            temp={}
            merged,groups=group_similar_clusters(merged,thr)
            i=0
            for label in merged:
                key=label[0]
                c=Cluster()
                temp[i]=c
                c.label=label[1]
                c.score=label[-1]
                c.number=i
                c.leaf=False
                for child in groups[key]:
                    c.children[child[0]]=d[child[0]]
                    c.files.extend(d[child[0]].files)
                c.label_frequency=label[2]
                c.sentences_count=label[4]
                c.token_count=label[3]
                c.total_sentences=label[5]
                c.threshold=thr
                c.level=level
                c.files.extend('Cluster_%d.txt'%key)
                i=i+1
            level=level+1
            d=temp
        
        croot=Cluster()
        croot.leaf=False
        croot.children=d
        croot.label="ROOT"
        croot.root=True
        croot.level=level
        return croot

    def __str__(self):
        s="Cluster: %d\n"%self.number
        s=s+"Label : %s\n"%self.label
        s=s+"Score : %s\n"%self.score
        s=s+"level : %d\n"%self.level
        s=s+"Threshold : %f\n"%self.threshold
        return s

    def display(self,indent=""):
        st=indent+"Cluster: %d Label: %s"%(self.number,self.label)
        print st
        if self.leaf:
            return
        children_values=sorted(self.children.values(),key = lambda x: -x.score)
        for child in children_values:
            child.display(indent=indent+" "*4) 

def list_parameters():
    s='''
1. business
2. category
3. city
4. state
5. models
    '''
    print s

def create_dataset_segregations(business=True):
    if business:
        print "Segregating Reviews by Businesses"
        os.system('python divide_by_business.py')
    f=open('../Dataset/business.json')
    city={}
    category={}
    state={}
    businesses=[]
    i=0
    print "\n\nSegregating businesses by city, state and category"
    for b in f:
        i=i+1
        sys.stdout.write('\rProgress: %0.1f '%(float(i)*100/61184)+"%")
        sys.stdout.flush()
        yb=YelpBusiness.parse_json(b)
        businesses.append((yb.business_id,yb.name))
        c=yb.city.lower().replace('/','_').replace(' ','_')
        if c not in city:
            city[c]=True
            fw=open('../BusinessReview/city_%s.txt'%c,'w')
            fw.close()
        fc=open('../BusinessReview/city_%s.txt'%c,'a')
        fc.write(yb.business_id+'\r\n')
        fc.close()
        st=yb.state.lower()
        if st not in state:
            state[st]=True
            fw=open('../BusinessReview/state_%s.txt'%st,'w')
            fw.close()
        fc=open('../BusinessReview/state_%s.txt'%st,'a')
        fc.write(yb.business_id+'\r\n')
        fc.close()
        for cat in yb.categories:
            cat=cat.lower().replace('/','_').replace(' ','_')
            if cat.lower() not in category:
                category[cat.lower()]=True
                fw=open('../BusinessReview/category_%s.txt'%cat,'w')
                fw.close()
            fc=open('../BusinessReview/category_%s.txt'%cat,'a')
            print i,st    
            fc.write(yb.business_id+'\r\n')
            fc.close()
    fcity=open('cities.pickle','w')
    pickle.dump(city.keys(),fcity)
    fcity.close()
    fbusiness=open('businesses.pickle','w')
    pickle.dump(businesses,fbusiness)
    fbusiness.close()
    fstate=open('states.pickle','w')
    pickle.dump(state.keys(),fstate)
    fstate.close()
    fcategory=open('categories.pickle','w')
    pickle.dump(category.keys(),fcategory)
    fcategory.close()
    f.close()    
    print "" 
    

def list_parameter_values(param):
    if param not in ['business','category','city','state','models']:
        print "Invalid category"
        return
    param_list={"business":businesses,"category":categories,"city":cities,"state":states,"models":trained}
    for param in param_list[param]:
        print param 

def plot(business_id,model,clusters,cluster_tree):
    labels=cluster_tree.get_labels_levelwise()
    #Plot level 0 top 9
    cluster_info={}
    cluster_num=[x.number for x in labels[0][0:9]]
    label_names=[x.label for x in labels[0][0:9]]
    cluster_num.append(-1)
    label_names.append('Others')
    
    for c in cluster_num:
        cluster_info[c]=([],[])
    for i in range(0,len(clusters)):
        if clusters[i] in cluster_num:
            key = clusters[i]
        else:
            key = -1
        cluster_info[key][0].append(model[i,0])
        cluster_info[key][1].append(model[i,1])
    colors=cm.rainbow(np.linspace(0,1,10))
    plots=[]
    fig=plt.figure()
    for i in range(0,len(colors[:-1])):
        plots.append(plt.scatter(cluster_info[cluster_num[i]][0],cluster_info[cluster_num[i]][1],color=colors[i],label=label_names[i]))
    plt.legend(plots,label_names[0:9])
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Top 9 Clusters reduced to 2 dimensions")
    fig.savefig('../Models/%s/scatter.jpg'%business_id)
    plt.show()    
   
 
def is_noun(x):
    return x.lower().strip() in nouns

def get_relative_similarity(a,b):
    x=wn.synset("%s.n.01"%a)
    y=wn.synset("%s.n.01"%b)
    return x.path_similarity(y)
    
def get_yelp_business(business_id):
    f=open('../Dataset/business.json')
    x=f.readline()
    while business_id not in x:
        x=f.readline()
    f.close()
    return YelpBusiness.parse_json(x)

def stem_tokens(tokens,stemmer):
    stemmed=[]
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    stemmed=[w for w in stemmed if len(w) > 3]
    return stemmed

def tokenize_and_stem(text):
    tokens=nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and len(w) > 3]
    stems=stem_tokens(tokens,stemmer)
    return stems

def tokenize_only(text):
    tokens=nltk.word_tokenize(text)
    tokens = [w for w in tokens if w.isalpha() and len(stemmer.stem(w)) > 3]
    return tokens

def get_vocab_mapping(dataset):
    vocab=[]
    stemmed=[]
    for i in dataset.keys():
        allwords_stemmed = tokenize_and_stem(dataset[i]) #for each item in 'synopses', tokenize/stem
        stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
        allwords_tokenized = tokenize_only(dataset[i])
        vocab.extend(allwords_tokenized)    
    
    vocab_frame = pd.DataFrame({'words': vocab}, index = stemmed)
    print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'
    print vocab_frame.head()
    return vocab_frame

def get_tfidf_matrix(dataset):
    vectorizer= TfidfVectorizer(tokenizer=tokenize_and_stem,stop_words="english",use_idf=True)
    model=vectorizer.fit_transform(dataset.values())
    return model,vectorizer.get_feature_names()

def get_tfidf_matrix_lsa(dataset,dimension_size=100):
    vectorizer= TfidfVectorizer(tokenizer=tokenize_and_stem,stop_words="english",use_idf=True)
    model=vectorizer.fit_transform(dataset.values())
    features=vectorizer.get_feature_names()
    lsa_matrix_gen=TruncatedSVD(n_components=dimension_size,random_state=42)
    model_fitted=lsa_matrix_gen.fit_transform(model)
    features=lsa_matrix_gen.components_
    return model_fitted,features 

def train_k_means(tfidf,K):
    km=KMeans(n_clusters=K)
    km.fit(tfidf)
    return km

def save_trained_model(business_id,obj,objname):
    if not os.path.exists('../Models/'):
        os.mkdir('../Models')
    if not os.path.exists('../Models/%s'%business_id):
        os.mkdir('../Models/%s'%business_id)
    fname='../Models/%s/%s.pickle'%(business_id,objname)
    f=open(fname,'w')
    pickle.dump(obj,f)
    f.close()

def segregate_by_cluster(business_id,K,dataset,cluster_list):
    if not os.path.exists('../Models/'):
        os.mkdir('../Models')
    if not os.path.exists('../Models/%s'%business_id):
        os.mkdir('../Models/%s'%business_id)
    if not os.path.exists('../Models/%s/Clusters'%business_id):
        os.mkdir('../Models/%s/Clusters'%business_id)
    base='../Models/%s/Clusters/'%business_id
    for i in range(0,K):
        f=open(base+'Cluster_%d.txt'%i,'w')
        f.close()
    l=len(cluster_list)
    keys=dataset.keys()
    for i in range(0,l):
        f=open(base+'Cluster_%d'%cluster_list[i],'a')
        f.write('%s\r\n'%dataset[keys[i]].encode('utf-8'))
        f.close()
    
def label_clusters(business_id,K,clusters):
    base='../Models/%s/Clusters/'%business_id
    sentence_count=FreqDist(clusters)
    total_sentences=len(clusters)
    labels=[]
    for i in range(0,K):
        f=open(base+'Cluster_%d'%i,'r')
        text=f.read().decode('utf-8')
        f.close()
        tokens=nltk.word_tokenize(text)
        tokens = [w for w in tokens if w.isalpha() and len(w) > 3 and w not in stopwords.words()]
        fd=FreqDist(tokens)
        frequent=fd.most_common(5)
        label="None"
        label_freq=0
        for f in frequent:
            if is_noun(f[0]):
                label,label_freq=f
                break
        
        relative_score=float(label_freq)/len(tokens)
        cluster_score=float(sentence_count[i])/total_sentences
        print "test label:",i,label
        labels.append((i,label,label_freq,len(tokens),sentence_count[i],total_sentences,relative_score*cluster_score))
    return labels

def load_model(business_id):
    base='../Models/%s/'%business_id
    files=get_filenames(business_id)
    dataset,business_id=select(files)
    if not os.path.exists('../Models/%s/'%business_id):
        print 'No training model found'
        return None,None,None,None
    km_trained=None
    try:
        km_trained=pickle.load(open(base+'kmeans.pickle'))
    except:
        print 'no trained model found'
        return dataset,None,None,None
    vector_space_model=None
    try:
        vector_space_model=pickle.load(open(base+'vsmodel.pickle'))
    except:
        print 'no trained model found'
        return None,None,None,None
    try:
        plottable_model=pickle.load(open(base+'plottable_model.pickle'))
    except:
        print 'no trained model found'
        return None,None,None,None
    return dataset,km_trained,vector_space_model,plottable_model

def group_similar_clusters(labels,threshold=0.3):
    K=len(labels)
    disjoint=[]
    for i in range(0,K):
        disjoint.append(i)

    for i in range(0,K):
        for j in range(i+1,K):
            rep1=disjoint[i]
            rep2=disjoint[j]
            if rep1!=rep2:
                sim=get_relative_similarity(labels[rep1][1],labels[rep2][1])
                if sim >= threshold:
                    if labels[rep1][-1] > labels[rep2][-1]:
                        new_label=rep1
                        other_label=rep2
                    else:
                        new_label=rep2
                        other_label=rep1
                    for k in range(0,j+1):
                        if disjoint[k] == other_label:
                            disjoint[k]=new_label
    d={}
    for i in range(0,K):
        if disjoint[i] not in d:
            d[disjoint[i]]=[]
        d[disjoint[i]].append((i,labels[i][1]))
    merged_clusters=[]
    i=0
    for k in d.keys():
        freq=0
        tokens=0
        sentences=0
        for x in d[k]:
            freq+=labels[x[0]][2]  
            tokens+=labels[x[0]][3]
            sentences+=labels[x[0]][4]
        score=float(freq)*float(sentences)/tokens/labels[k][5]
        merged=(k,labels[k][1],freq,tokens,sentences,labels[k][5],score)
        merged_clusters.append(merged)     
        i=i+1
    return merged_clusters,d


def load_labels(business_id):
    base='../Models/%s/'%business_id
    labels_unmerged=None
    cluster_tree=None
    if not os.path.exists('../Models/%s/'%business_id):
        print "No Data of Labels Exist"
        return None,None
    try:
        f=open(base+'labels_level_0.pickle')
    except:
        print "Labels have not been generated for this business"
        return None,None
    labels_unmerged=pickle.load(f)
    try:
        f=open(base+'cluster_tree.pickle')
    except:
        print "Labels have not been generated for this business"
        return None,None
    cluster_tree=pickle.load(f)
    return labels_unmerged,cluster_tree 

def pre_process(fname):
    if not os.path.exists(fname):
        print 'File Not Found'
        return
    business_id=fname.split('/')[-1].split('.')[0]
    directory=fname[0:len(fname)-len(business_id)-5]
    business=get_yelp_business(business_id)
    dataset=business.update_review_data(root_dir=directory)
    return dataset,business_id 

def select(fname):
    for name in fname:
        if not os.path.exists(name):
            print 'File Not Found',fname
            return
    x=set([y[0] for y in businesses])
    s=[]
    c=[]
    ca=[]
    format_id=""
    for name in fname:
        filename=name.split('/')[-1]
        directory=name[0:len(name)-len(filename)]
        if filename.startswith('city') or filename.startswith('category') or filename.startswith('state'):
            f=open(name)
            x.intersection_update(set(f.read().splitlines()))
            f.close() 
            if filename.startswith("city"):
                c.append(filename[:-4][5:])
            if filename.startswith("state"):
                s.append(filename[:-4][6:])
            if filename.startswith("category"):
                ca.append(filename[:-4][9:])
        else:
            x.intersection_update(set([filename[:-5]]))
            format_id=filename[:-5]
    dataset={}
    c.sort()
    s.sort()    
    ca.sort()
    if len(format_id)==0:
        separator=""
        if s:
            format_id+=format_id+"s"
            for st in s:
                format_id+="#%s"%st
            separator="@"    
        if c:
            format_id+=separator+"c"
            for ci in c:
                format_id+="#%s"%ci
            separator="@"
        if ca:
            format_id+=separator+"ca"
            for cat in ca:
                format_id+="#%s"%cat
    businesslist=list(x)
    total_businesses=len(businesslist)
    i=0
    for business in businesslist:
        i=i+1
        sys.stdout.write('\r Reading Business Reviews: %d/%d'%(i,total_businesses))
        sys.stdout.flush()
        dataset_temp,business_id=pre_process(directory+"%s.json"%business.strip())
        dataset.update(dataset_temp)
    print ""
    return dataset,format_id

def get_filenames(format_id,directory="../BusinessReview/"):
    if "#" not in format_id:
        return [directory+business_id+".json"]
    x=format_id.split('@')
    files=[]
    for sec in x:
        sec=sec.split("#")
        if sec[0] == "s":
            prefix="state_"
        elif sec[0]== "c":
            prefix="city_"
        elif sec[0]=="ca":
            prefix="category_"
        for name in sec[1:]:
            files.append(directory+prefix+name+".txt")
    return files

def train(business_id,dataset,num_of_clusters,lsa=False,lazy=False):
    print "Finding K Clusters for Business ID: %s and K = %d"%(business_id,num_of_clusters)        
    print 'Performing Tokenization, Stemming and Stopword Removal followed by Conversion to Vector Space Model'
    if lsa:
        vector_space_model,terms = get_tfidf_matrix_lsa(dataset)
    else:
        vector_space_model,terms = get_tfidf_matrix(dataset)
    plottable_model,temp=get_tfidf_matrix_lsa(dataset,2)
    print 'TF-IDF Matrix Generated. Performing K-Means Clustering.....'
    save_trained_model(business_id,vector_space_model,'vsmodel')
    save_trained_model(business_id,plottable_model,'plottable_model')
    km_trained = train_k_means(vector_space_model,num_of_clusters)
    save_trained_model(business_id,km_trained,'kmeans')
    clusters = km_trained.labels_.tolist()
    print "Clusters Trained"
    df=pd.DataFrame(zip(dataset.keys(),dataset.values(),clusters),columns=["SentenceID","Sentence","Cluster#"],index=clusters)
    train_information={"K":num_of_clusters,"business_id":business_id,"lsa":lsa}
    save_trained_model(business_id,train_information,'last_train_info')
    refresh_trained()
    return km_trained,clusters,df

def analyze(business_id,dataset,clusters,segregate=True):
    base='../Models/%s/'%business_id
    if not os.path.exists('../Models/%s/'%business_id):
        print 'No training model found'   
        return
    f=open(base+"last_train_info.pickle")
    trained_info=pickle.load(f)
    f.close()
    num_of_clusters=trained_info["K"] 
    if segregate:
        print 'Segregating text in the dataset into clusters'
        segregate_by_cluster(business_id,num_of_clusters,dataset,clusters)
    print 'Generating labels for generated clusters'
    labels=label_clusters(business_id,num_of_clusters,clusters)
    save_trained_model(business_id,labels,"labels_level_0")
    print 'Merging clusters on the basis of semantic similarity of words'
    cluster_tree=Cluster.build(labels)
    save_trained_model(business_id,cluster_tree,"cluster_tree") 
    return cluster_tree

def view(business_id):
    labels,cluster_tree=load_labels(business_id)    
    if not labels:
        return
    print 'Hierarchical Display of Clusters'
    cluster_tree.display()
    levelwise=cluster_tree.get_labels_levelwise()
    print 'Level Wise Display of Clusters in decreasing order of score'    
    for i in range(0,len(levelwise)):
        print "#####################Level-%d : %d Clusters ####################"%(i,len(levelwise[i]))    
        data=[(x.number,x.label,x.score) for x  in levelwise[i]]
        df=pd.DataFrame(data,columns=["Cluster#","Label","Score"])
        print df
    dataset,km,vsm,plm=load_model(business_id)
    clusters=km.labels_.tolist()
    plot(business_id,plm,clusters,cluster_tree)
    
    

def main(fname,num_of_clusters):
    if not os.path.exists(fname):
        print 'File Not Found'
        return
    num_of_clusters=int(num_of_clusters)
    business_id=fname.split('/')[-1].split('.')[0]
    print business_id    
    business=get_yelp_business(business_id)
    dataset=business.update_review_data()

    vocab=get_vocab_mapping(dataset)

    vector_space_model,terms = get_tfidf_matrix(dataset)
    
    save_trained_model(business_id,vector_space_model,'vsmodel')
 
    km_trained = train_k_means(vector_space_model,num_of_clusters)
    
    save_trained_model(business_id,km_trained,'kmeans')
    
    clusters = km_trained.labels_.tolist()

    segregate_by_cluster(business_id,num_of_clusters,dataset,clusters)

    labels=label_clusters(business_id,num_of_clusters,clusters)         
   
    save_trained_model(business_id,labels,"labels_level_0")
    merged,grouping = group_similar_clusters(labels,)
    labels=sorted(labels,key = lambda k: -k[6])



    print ("Cluster#","Label","Label Frequency","Total Tokens","Cluster Count","Total Sentences","Score")
    
    for label in labels:
        print label

    merged,grouping = group_similar_clusters(labels,0.15)   

    save_trained_model(business_id,merged,"merged")
    save_trained_model(business_id,grouping,"groups")
    merged_sorted=   sorted(merged,key = lambda k: -k[6]) 
    
    for grp in grouping:
        print grouping[grp]

    for label in merged_sorted:
        print label

def show_usage():
    usage=''' 
 HOW TO USE: 
 ##############################################################################################################################
 Convert the dataset into usable form; The zip file should be extracted and the files should be renamed
 The json files should be placed in the ../Dataset folder. e.g yelp_datase...._business.json should be
 renamed to business.json
 Usage:
 python cluster_yelp.py build
 -------------------------------------
 List available parameters;
 Usage:   
 python cluster_yelp.py list   
 -------------------------------------
 List values for a given parameter;      
 Usage:   
 python cluster_yelp.py list <parameter_name>
    parameter_name : One of  {city,state,category,business,models}. Models lists the trained models;
 -------------------------------------
 Train reviews of a business for K number of clusters;
 Usage:       
 python cluster_yelp.py train <filename> <cluster_size> <lsa=False>
    filename: Paths to the file with segregations separated by comma, the intersection of businesses is chosen (i.e. AND operation);
    cluster_size : An integer > 0;   
    lsa : Perform latent semantic analysis before clustering default= False;
 e.g. python cluster_yelp.py train '../BusinessReview/city_phoenix.txt,../BusinessReview/category_doctors.txt' 100
 -------------------------------------
 Analyze the data of already trained information. This includes segregation into clusters, 
 labelling of clusters, hierarchical merging on the basis of semantic similarity of cluster labels.
 Usage:
 python cluster_yelp.py analyze <model_id> <segregate=True>
    model_id: The model_id for training has been already performed and needs to be analyzed;
    segregate: OPTIONAL Perform segregation of dataset into clusters. Should be set to False if segregation is performed already;
 -------------------------------------
 View the clusters hierarchically and levelwise ranked on the basis of their score.
 Usage:
 python cluster_yelp.py view <model_id>
    model_id: The model_id for training has been already performed and analyzed and to be viewed;  
 -------------------------------------
 Perform all operations from training to viewing
 Usage:       
 python cluster_yelp.py complete <filename> <cluster_size> <lsa=False>
    filename: Paths to the file with segregations separated by comma, the intersection of businesses is chosen (i.e. AND operation); 
    cluster_size : An integer > 0;   
    lsa : Perform latent semantic analysis before clustering default= False;
 e.g. python cluster_yelp.py complete '../BusinessReview/city_phoenix.txt,../BusinessReview/category_doctors.txt' 100
 ##############################################################################################################################    
    '''
    print usage

if __name__ == "__main__":
    #options = train, analyze, view, all
    print "#########Yelp Review Clustering Tool#########"
    argc=len(sys.argv)
    if argc == 1:
        show_usage()
    elif sys.argv[1] == "build":
        decision=raw_input("Do you want to segregate reviews into businesses. Necessary step if not done before.[y/n]\n")
        create_dataset_segregations("y" in decision.lower())
    elif sys.argv[1] == "list":
        if argc==2:
            list_parameters()
        elif argc >= 2:
            list_parameter_values(sys.argv[2])
    elif sys.argv[1].strip().lower() == "train":
        filename=""
        num_of_clusters=0
        lsa=False
        lazy=False
        if argc >= 4:
            filename=sys.argv[2].split(',')
           
            num_of_clusters=int(sys.argv[3])
            if argc >= 5:
                lsa=sys.argv[4].lower().strip() == "true"
            if argc >= 6:
                lazy=sys.argv[5].lower().strip() == "true"
            dataset,business_id=select(filename)
            if dataset:
                km,cl,df=train(business_id,dataset,num_of_clusters,lsa,lazy)
                print df
            else:
                print "Not Enough Data !!!"
        else:
            show_usage()

    elif sys.argv[1].strip().lower() == "analyze":
        business_id=""
        segregate=True
        if argc >= 3:
            business_id=sys.argv[2]
            if argc == 4:
                segregate = sys.argv[3].lower().strip() == "true"    
            dataset,km,vsmodel,plm=load_model(business_id)
            if km:
                analyze(business_id,dataset,km.labels_.tolist(),segregate)    
        else:
            show_usage()
    elif sys.argv[1].strip().lower() == "view":
        if argc > 2:
            business_id=sys.argv[2]
            view(business_id)
        else:
            show_usage()
     
    elif sys.argv[1].strip().lower() == "complete":
        filename=""
        num_of_clusters=0
        lsa=False
        lazy=False
        if argc >= 4:
            filename=sys.argv[2].split(',')
            num_of_clusters=int(sys.argv[3])
            if argc >= 5:
                lsa=sys.argv[4].lower().strip() == "true"
            if argc >= 6:
                lazy=sys.argv[5].lower().strip() == "true"

            dataset,business_id=select(filename)
            if dataset:
                km,cl,df=train(business_id,dataset,num_of_clusters,lsa,lazy)
                print df
                analyze(business_id,dataset,cl)
                view(business_id)
            else:
                print "Not Enough Data !!!"
        else:
            show_usage()
           

    #main(sys.argv[1],sys.argv[2]) 
    
