#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
print len(enron_data["SKILLING JEFFREY K"])
print len( [1 for person in enron_data if enron_data[person]['poi'] ] )
print enron_data["PRENTICE JAMES"]["total_stock_value"]
print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
execs = [s for s in enron_data.keys() if ("SKILLING" in s) or ("LAY" in s) or ("FASTOW" in s) ]
print max( [(enron_data[person]['total_payments'],person) for person in execs] )
#print max(enron_data["SKILLING JEFFREY K"]["total_payments"],enron_data["LAY KENNETH L"]["total_payments"],enron_data["FASTOW ANDREW S"]["total_payments"])
print len([enron_data[person]['salary'] for person in enron_data if enron_data[person]['salary'] != 'NaN']) # 95
print len([enron_data[person]['email_address'] for person in enron_data if enron_data[person]['email_address'] != 'NaN' ]) # 111

print float(len([enron_data[person]['total_payments'] for person in enron_data if enron_data[person]['total_payments'] == 'NaN' ]) ) / float( len( enron_data.keys() ) )*100.

print len( [1 for person in enron_data if enron_data[person]['poi'] and enron_data[person]['total_payments'] == 'NaN' ] )

print len( [1 for person in enron_data if enron_data[person]['total_payments'] == 'NaN' ] )
