import numpy as np
import math
import sys,os
import re
import cPickle as pickle

stop_words = pickle.load( open('stop_words.dat') )
all_files = {}
all_terms = {}
IDF = {}

def filter_terms():

    global all_terms
    all_terms = { t: all_terms[t] for t in all_terms if all_terms[t]>10 and t not in stop_words}
    print 'Totolly we have' , len(all_terms), ' terms after filtering'

def get_words_bag( tokens ):

    bag={}

    for w in tokens:
        bag[w]=bag.get(w,0)+1

    return bag

def get_all_terms():

    global all_terms,all_files

    if 'all_terms.dat' in os.listdir(os.getcwd()):
        print 'All terms has already been got.'
    else:
        for filename, tokens in all_files.items():
            for k in tokens:
                all_terms[k] = all_terms.get(k,0) + tokens[k]

        filter_terms()

        fout = open('all_terms.dat','w')
        pickle.dump(all_terms,fout)
        fout.close()

        print 'All terms list got.'

    all_terms = pickle.load(open('all_terms.dat','r'))

def get_all_files():

    global all_files

    if 'all_files.dat' in os.listdir(os.getcwd()):
        print 'File data has already been got.'

    else:
        
        for subset in ['s1','s2','s3','s4','s5']:
            for classname in ['baseball', 'hockey']:

                path =  os.path.join('.','data 1', subset, classname) 
                print path

                for file_name in os.listdir( path ):
                    full_filename = os.path.join(path,file_name) 
                    fin = open(full_filename).read()
                    fin = fin[fin.index('\n'):]
                    all_files[full_filename] = get_words_bag(re.findall('[a-zA-Z\']+',fin))

        fout = open('all_files.dat','w')
        pickle.dump(all_files,fout)
        fout.close()

        print 'All file data got.'

    all_files = pickle.load(open('all_files.dat','r'))

def get_idf():

    global IDF,all_terms

    if 'IDF.dat' in os.listdir(os.getcwd()):
        print 'IDF has already been got.'
    else:
        tot_docs = len(all_files)

        for t in all_terms:

            num_with_t = sum(  t in all_files[filename] for filename in all_files )

            IDF[t]=math.log(tot_docs / float(num_with_t))

        fout = open('IDF.dat','w')
        pickle.dump(IDF,fout)
        fout.close()
        print 'IDF data got.'

    IDF = pickle.load(open('IDF.dat','r'))

def get_tfidf():
    
    if 'TF-IDF.dat' in os.listdir(os.getcwd()):
        print 'TF-IDF has already been got.'
    else:
        N=len(all_files)
        D=len(all_terms)

        X , y = np.zeros(shape = (N,D)), np.zeros(N)

        for i,filename in enumerate(sorted(all_files)):
            if 'hockey' in filename:
                y[i] = 1.0

        for i,filename in enumerate(sorted(all_files)):
            tokens = all_files[filename]

            tot = float(len(tokens))

            for j,term in enumerate(sorted(all_terms)):
                X[i,j] = tokens.get(term, 0.0) / tot * IDF[term]

        fout = open('TF-IDF.dat','w')
        pickle.dump((X,y),fout)
        fout.close()
        print 'TF-IDF data got.'
        
def main():

    global all_files , stop_words , IDF, all_terms

    get_all_files()
    get_all_terms()
    get_idf()
    get_tfidf()

    #get_word_index()

if __name__ == '__main__':

    main()
