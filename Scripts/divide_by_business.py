import json
import sys
import os

'''
    This script is responsible for segregating the review by business
    A new folder BusinessReview is created in which reviews are segregated into
    files of name <business_id>.json
'''

#Create a business review directory if it does not exist
if not os.path.exists('../BusinessReview'):
    os.mkdir('../BusinessReview')


def digest_business():
    '''
        Returns a dictionary of businesses with business_names
    '''
    business_type_dict={}
    decoder=json.JSONDecoder()
    fbusiness=open('../Dataset/business.json','r')

    for business in fbusiness:

        business_attr=decoder.decode(business)
        business_id=business_attr['business_id']
        business_name=business_attr['name']
        business_type_dict[business_id]=business_name

    fbusiness.close()
    return business_type_dict

def get_file_list(business_type_dict):
    '''
        Creates an empty file for each business.
    '''
    files_dict={}
    for key,value in business_type_dict.iteritems():
        files_dict[key]=open('../BusinessReview/%s.json'%key,'w')
        files_dict[key].close()
    return

def dump_reviews():
    '''
        Dumps the reviews for each business in their respective file.
    '''
    decoder=json.JSONDecoder()
    review=open('../Dataset/review.json','r')
    i=1
    for line in review:
        sys.stdout.write('\rProgress: %0.1f '%(float(i)*100/1569264)+"%")
        sys.stdout.flush()
        review_attr=decoder.decode(line)
        business_id=review_attr['business_id']
        dummy_file=open('../BusinessReview/%s.json'%business_id,'a')
        dummy_file.write(line)
        dummy_file.close()
        i=i+1
    review.close()


def main():
    '''
        Main part of the script
        1. Get business list
        2. Create empty files for each business
        3. Dump reviews into their respective business file
    '''
    business_type_dict=digest_business()
    files_dict = get_file_list(business_type_dict)
    dump_reviews()

if __name__ == '__main__':
    main()
