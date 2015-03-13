import numpy as np
import sys,os
import re
import cPickle as pickle

stop_words = set(['of','s','t'])

def get_word_index():
    
    words = pickle.load(open('words_bag.dat','r'))

    sorted_words=sorted(words)
    open('tmp','w').write(str( sorted_words ) )

    index = { w:k for k,w in enumerate(sorted_words)}

    return index

def get_subset_feature():
    
    for subset in ['s1','s2','s3','s4','s5']:
        for classname in ['baseball', 'hockey']:

            path =  os.path.join('.','data 1', subset, classname) 
            print path

            for file_name in os.listdir( path ):

                fin = open(os.path.join(path,file_name)).read()
                
                for w in re.findall('[a-zA-Z0-9]+',fin):
                    #dic[i] = dic.get(i,0)+1 
                    words.add(w)

def get_words_bag():

    if 'words_bag.dat' in os.listdir(os.getcwd()):
        print 'already got.'
        return 

    words = set()
    
    for subset in ['s1','s2','s3','s4','s5']:
        for classname in ['baseball', 'hockey']:

            path =  os.path.join('.','data 1', subset, classname) 
            print path

            for file_name in os.listdir( path ):

                fin = open(os.path.join(path,file_name)).read()
                
                for w in re.findall('[a-zA-Z]+',fin):
                    #dic[i] = dic.get(i,0)+1 
                    words.add(w)

    fout = open('words_bag.dat','w')
    pickle.dump(words,fout)

def main():

    get_words_bag()
    get_word_index()

if __name__ == '__main__':

    main()
