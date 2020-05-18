from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings
import speech_recognition as sr



warnings.filterwarnings('ignore')
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)
article=Article('https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521')
article.download()
article.parse()
article.nlp()
corpus=article.text
#print(corpus)
 
text=corpus
sent_tokens=nltk.sent_tokenize(text)#convert the text into a alist of sentences
#print(sent_tokens)

#creatre a dictionary (key:value) pair to remove punctuations
remove_punct_dict=dict( (ord(punct),None) for punct in string.punctuation)
#print(string.punctuation)
#print(remove_punct_dict)

#create ala function to return a list of lenmatized lowercase words after removing puctuatuins.i,e all the sentences in the article are now converted into a list
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))
#prints the tokenozation text by removing the punctuation

#print(LemNormalize(text))

#keyword matching
#GREETINGS INPUT
GREETING_INPUTS=["hi","hello","hola","greetings","wassup","hey"]
#greeting response back
GREETING_RESPONSE=["howdy","hi","hey","what's good","hello"]
#function to return a random greeting response
def greeting(sentence):
    #return a randomly choosen responce
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSE)




#generate the respnse to the given question
def responce(user_responce):
    
#the user's query is taken        
    #user_responce='what is chronic kidney disease'
#the user may give his input as capitals so we should convert them into lower()
    user_responce=user_responce.lower()
#set the chat bot respnse to an empt srting i.e declare the roborespnse as a string
    robo_responce=''
#convert the user_responce into a list
    sent_tokens.append(user_responce)
#create a TfidVectorizer object it is used to know how man tomes a word has occured
    TfidVec=TfidfVectorizer(tokenizer=LemNormalize,stop_words='english')
#convert the text into a matrix of TF-IDF features
    tfidf=TfidVec.fit_transform(sent_tokens)
#print(tfidf)

#get the measure of similarity(similarit scores)
    vals=cosine_similarity(tfidf[-1],tfidf)
#print(vals)
#get the index of the most similar text/sentence to the user response
    idx=vals.argsort()[0][-2]

    #reduce the domensionalit of vals
    flat=vals.flatten()
#sort the list in asc
    flat.sort()
#get the most simliar score for the user's responce
    score=flat[-2]


#print the similarit score
#print(score) 
#if the score is 0 then the most similar score to the user resoponce
    if(score==0):
        robo_responce=robo_responce+"i aplogise i didn't understand"
    else:
        robo_responce=robo_responce+sent_tokens[idx]
    
#pritn the chat bot respnce
    #print(robo_responce)
    sent_tokens.remove(user_responce)
    return robo_responce







r=sr.Recognizer()
with sr.Microphone() as source:


    flag=True
    print("BOT:Iam doctor bot and iam going to answeer your questions")
    while(flag==True):
        print("speak:")
        audio=r.listen(source)
        try:
            text=r.recognize_google(audio)
            print("you said:{}".format(text))
            user_responce=text
            if(user_responce!='bye'):
                if(user_responce=='thanks' or user_responce=='thank you'):
                    flag=False
                    print("BOT:you are welcome")
                else:
                    if(greeting(user_responce)!=None):
                        print("BOT:"+greeting(user_responce))
                    else:
                        print("BOT: "+responce(user_responce))
                
            else:
                flag=False
                print("BOT:chat with u later")
        except:
            print("could not recognize")
        