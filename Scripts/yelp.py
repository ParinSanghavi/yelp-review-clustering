import json
import string
import re 
import os

class YelpBusiness(object):

    def __init__(self):
        self._id=''
        self.reviews_data={}
        self.reviews={}
        self.categories=[]
        self.location=''

    @classmethod
    def parse_json(cls,json_string):
        j=json.JSONDecoder()
        parsed=j.decode(json_string)
        yb=YelpBusiness()
        for key in parsed.keys():
            yb.__setattr__(key,parsed[key])
        return yb

    def update_review_data(self,root_dir='../BusinessReview/'):
        filename=os.path.join(root_dir,self.business_id+".json")
        if  not os.path.exists(filename):
            print 'Reviews file does not exist'
            return {}
        else:
            reviews=open(filename)
            for review in reviews:
                r=YelpReview.parse_json(review)
                self.reviews[r.review_id]=r
                self.reviews_data.update(r.tokenize_sentences())
        return self.reviews_data
         
    def get_review_texts(self,root_dir='../BusinessReview/'):
        filename=os.path.join(root_dir,self.business_id+".json")
        print filename
        if  not os.path.exists(filename):
            print 'Reviews file does not exist'
            return {}
        else:
            reviews=open(filename)
            t={}
            for review in reviews:
                r=YelpReview.parse_json(review)
                self.reviews[r.review_id]=r
                t[r.review_id]=r.text
        return t        


class YelpReview(object):

    def __init__(self):
        self._id=''
        self.text=''
        self.stars=''
        self.business_id=''

    @classmethod
    def parse_json(cls,json_string):
        j=json.JSONDecoder()
        parsed=j.decode(json_string)
        yr=YelpReview()
        for key in parsed.keys():
            yr.__setattr__(key,parsed[key])
        yr.text=yr.text.lower()
        yr.text=yr.text.replace('\n',' ')
        return yr

    def tokenize_sentences(self):
        pattern=r'[.?]|\s+and\s+'
        sents = re.split(pattern,self.text)
        sents = [ s for s in sents if s.strip()]
        sents_dict={}
        i=1
        for s in sents:
            sents_dict[self.review_id+'_'+str(i)]=s
            i=i+1
        return sents_dict 
        


