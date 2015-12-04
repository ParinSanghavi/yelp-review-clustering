import json
import sys
import os

if not os.path.exists('../BusinessReview'):
    os.mkdir('../BusinessReview')


def digest_business():
    
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
    files_dict={}
    for key,value in business_type_dict.iteritems():
        files_dict[key]=open('../BusinessReview/%s.json'%key,'w')
        files_dict[key].close()
    return

def dump_reviews():
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
    business_type_dict=digest_business()
    files_dict = get_file_list(business_type_dict)
    dump_reviews()



if __name__ == '__main__':
    main()
