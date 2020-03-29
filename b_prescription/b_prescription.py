#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 17:32:44 2020

@author: hteza
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# dataframe import
df = pd.read_csv("/Users/hteza/Desktop/Class/RADS602/Apriori_Algorithm_Ass/prescriptionDB.csv")

# creating a seperate basket, with specific codes
# since the question specifies Code2=OMPZ
basket=pd.get_dummies(df, prefix=['1', '2', '3', '4'], columns=['Code.1', 'Code.2','Code.3','Code.4']).set_index("Item ID")

# creating frequent itemsets
# mimimum support = 0.001
frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True) 

# mimumum confidence = 0.5
rules = association_rules(frequent_itemsets,metric="confidence",min_threshold=0.5)

# to get the count of transaction count > 5
# refinding the number of times the relationship occured 
# by multiplying the support with the total number of transactions 

# rules["no_of_transaction_for_antecedent"]=rules["antecedent support"].apply(lambda x: x*len(basket.index))
# rules["no_of_transaction_for_consequents"]=rules["consequent support"].apply(lambda x: x*len(basket.index))
rules["no_of_transaction_for_relationship"]=rules["support"].apply(lambda x: x*len(basket.index))

# final rule
# Code2 = OMPZ
# count>5
filter_rules=rules[(rules["consequents"]=={'2_OMPZ'}) & (rules["no_of_transaction_for_relationship"]>5)]