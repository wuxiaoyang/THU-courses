import numpy as np
import sys, os
import re

def get_sample(s):

    label , feature = int(s[0:3]) , s[1:]

    for item in re.findall('(\d+):(\d+\.?\d*)', feature):
        item[0],item[1]

def read_data():

    try:
        feature_file=open('feature_example_s1','r').readlines()
        get_sample(feature_file[0])
        
    except:
        pass

def main():
    read_data()

if __name__ == '__main__':
    main()
