import json
import string
import re 
import os

class YelpBusiness(object):

    '''
        Represents a Yelp Business
        Attribute list:
        business_id : Unique Business ID;
        full_address : Address of the business;
        hours: timings on different days of thw week;i
        categories : A list of categories business appears in;
        city: City;
        state: State;
        name: Name of the business
        review_count: No. of reviews for this business
        stars: Star rating;
        attributes: Custom attributes;
        type: business always;
        latitude: Latitude;
        logitude: Longitude;
    '''
    def __init__(self):
        '''
            Initialize a Yelp business
        '''
        #a dictionary of all of business reviews broken into sentence fragments
        self.reviews_data={}
        #Dictionary of Yelp Review objects
        self.reviews={}

    @classmethod
    def parse_json(cls,json_string):
        '''
            Given a json string returns a YelpBusiness object
            Sample json string: 
            ---------------
            {"business_id": "vcNAWiLM4dR7D2nwwJ7nCA", "full_address": "4840 E Indian School Rd\nSte 101\nPhoenix, AZ 85018",
             "hours": {"Tuesday": {"close": "17:00", "open": "08:00"}, "Friday": {"close": "17:00", "open": "08:00"},
             "Monday": {"close": "17:00", "open": "08:00"}, "Wednesday": {"close": "17:00", "open": "08:00"}, 
             "Thursday": {"close": "17:00", "open": "08:00"}}, "open": true, "categories": ["Doctors", "Health & Medical"],
             "city": "Phoenix", "review_count": 9, "name": "Eric Goldberg, MD", "neighborhoods": [], 
             "longitude": -111.98375799999999, "state": "AZ", "stars": 3.5, "latitude": 33.499313000000001, 
             "attributes": {"By Appointment Only": true}, "type": "business"}
            ---------------
        '''
        j=json.JSONDecoder()
        parsed=j.decode(json_string)
        yb=YelpBusiness()
        for key in parsed.keys():
            yb.__setattr__(key,parsed[key])
        return yb

    def update_review_data(self,root_dir='../BusinessReview/'):
        '''
            Adds the reviews associated with this business into self.reviews
            Also compiles all the reviews in the form of a dictionary of broken sentences
            into self.reviews_data
            Returns a dictionary of all reviews of the business broken into sentence fragments.
        '''
        filename=os.path.join(root_dir,self.business_id+".json")
        if  not os.path.exists(filename):
            print 'Reviews file does not exist'
            return {}
        else:
            reviews=open(filename)
            for review in reviews:
                r=YelpReview.parse_json(review) #Create a yelp review object and add it to self.reviews
                self.reviews[r.review_id]=r
                # Break the review into sentences and add it self.reviews_data
                self.reviews_data.update(r.tokenize_sentences())
        return self.reviews_data
         
    def get_review_texts(self,root_dir='../BusinessReview/'):
        '''
            Returns a dictionary of reviews in the form of review texts without breaking into sentences.
        '''
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
    '''
        Represents a Yelp review object
        Attributes:
        votes: Review votes;
        user_id: The user id who gave the review;
        review_id: Unique ID of the review;
        text: Actual review text;
        type: review always;
        business_id: The business_id of the business to which the review belongs;
    '''

    def __init__(self):
        '''
            Initialize a YelpReview object
        '''
        self.text=''

    @classmethod
    def parse_json(cls,json_string):
        '''
            Returns a YelpReview object by parsing JSON string from the dataset
            Sample JSON string:
            -------
            {"votes": {"funny": 0, "useful": 2, "cool": 1}, "user_id": "Xqd0DzHaiyRqVH3WRG7hzg",
             "review_id": "15SdjuK7DmYqUAj6rjGowg", "stars": 5, "date": "2007-05-17", 
             "text": "dr. goldberg offers everything i look for in a general practitioner.
              he's nice and easy to talk to without being patronizing; he's always on time
              in seeing his patients; he's affiliated with a top-notch hospital (nyu) which 
              my parents have explained to me is very important in case something happens and you
              need surgery; and you can get referrals to see specialists without having to see him
              first.  really, what more do you need?  i'm sitting here trying to think of any 
              complaints i have about him, but i'm really drawing a blank.", "type": "review",
              "business_id": "vcNAWiLM4dR7D2nwwJ7nCA"}
            ------
        '''
        j=json.JSONDecoder()
        parsed=j.decode(json_string)
        yr=YelpReview()
        for key in parsed.keys():
            yr.__setattr__(key,parsed[key])
        yr.text=yr.text.lower()
        yr.text=yr.text.replace('\n',' ')
        return yr

    def tokenize_sentences(self):
        '''
            Breaks the set into sentence fragments by using regular expression
            Returns a dictionary with key: review_id+number and value a sentence fragment
        '''
        # Regular expression pattern: Split by either "." , "?" or " and "
        pattern=r'[.?]|\s+and\s+'
        sents = re.split(pattern,self.text)
        # Remove empty strings from the list
        sents = [ s for s in sents if s.strip()]
        sents_dict={}
        i=1
        for s in sents:
            # key = review_id + '_' + enumerator value "i" 
            sents_dict[self.review_id+'_'+str(i)]=s
            i=i+1
        return sents_dict 
        


