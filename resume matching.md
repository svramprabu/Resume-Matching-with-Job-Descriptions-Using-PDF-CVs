# Imports


```python
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertModel
import torch

# import transformers
```

# NLP and cleaning fn


```python
nlp = spacy.load('en_core_web_sm')

stopwords = ENGLISH_STOP_WORDS
# lemmatizer = WordNetLemmatizer()

def clean(doc):
    text_no_namedentities = []
    document = nlp(doc)
    ents = [e.text for e in document.ents]
    for item in document:
        if item.text in ents:
            pass
        else:
            text_no_namedentities.append(item.text)
    doc = (" ".join(text_no_namedentities))

    doc = doc.lower().strip()
    doc = doc.replace("</br>", " ")
    doc = doc.replace("-", " ")
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    doc = " ".join([token for token in doc.split() if token not in stopwords])
    # doc = "".join([lemmatizer.lemmatize(word) for word in doc])
    doc=nlp(doc)
    doc = " ".join([word.lemma_ for word in doc])
    doc=" ".join(set((doc.split())))
    
    return doc
```

# Embedding fn


```python
def embedding(X_train):
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = DistilBertModel.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # X_train = job_description['job_description'].to_list()
    # print(type(X_train))

    res = tokenizer(X_train,padding=True, truncation=True, max_length=3, return_tensors="pt")
    # print(res)
    # len(res)

    with torch.no_grad():
        outputs = model(**res)
        # outputs = model(res['input_ids'])
        # model?
        JD_vects=outputs.last_hidden_state
    return JD_vects.reshape(1,-1)
    # return JD_vects[0].tolist()

```

# Cosine Similarity Function


```python
def cosim(df1,df2):
    return cosine_similarity(df1,df2)
```

# import resume and jd csv files


```python
resume_data = pd.read_csv("resume_data.csv",index_col=0)
```


```python
# job_description = pd.read_csv('training_data.csv', chunksize=15)
job_description = pd.read_csv('training_data.csv')
job_description=job_description.head(15)
```

# clean resume and jd text


```python
resume_data['Cleaned_Resume']=resume_data['resume_text'].apply(lambda x:clean(x))
```


```python
job_description['Cleaned_JD']=job_description['job_description'].apply(lambda x:clean(x))
```

# embed resume and jd text


```python
resume_data['Cleaned_Resume_vector']=resume_data['Cleaned_Resume'].apply(lambda x:embedding(x))
```


```python
job_description['Cleaned_JD_vector']=job_description['Cleaned_JD'].apply(lambda x:embedding(x))
```

# resume matching by cosine similarity 


```python

choice = int(input("Enter index of a job decription to choose from 0-14:\n"))

if choice <= 14 and choice > 0:
    # job_description['company_name']
    print(f"\nYour choice of JD is from {job_description[job_description['company_name'].index==choice]['company_name'].values[0]}\n")
    print(f"Job Description:\n\n {job_description[job_description['company_name'].index==choice]['job_description'].values[0]}\n")
    print(f"Position for which resumes are shortlisted: {job_description[job_description['company_name'].index==choice]['position_title'].values[0]}\n")

    resume_data['jd_vector_of_choice'] = resume_data['Cleaned_Resume_vector'].apply(lambda x: cosim(x,job_description['Cleaned_JD_vector'][choice].tolist()))
    print("Shortlisted resume File names are as below\n")
    df=(resume_data.sort_values('jd_vector_of_choice', ascending=False)['File_name'].head())    
    print(df)
    print()
else:
    print("Invalid choice try again")
    


```

    Enter index of a job decription to choose from 0-14:
     1


    
    Your choice of JD is from Apple
    
    Job Description:
    
     description
    as an asc you will be highly influential in growing mind and market share of apple products while building longterm relationships with those who share your passion 
    customer experiences are driven through you and your partner team growing in an ever changing and challenging environment you strive for perfection whether its maintaining visual merchandising or helping to grow and develop your partner team
    
    qualifications
    a passion to help people understand how apple products can enrich their livesexcellent communication skills allowing you to be as comfortable in front of a small group as you are speaking with individuals years preferred working in a dynamic sales andor results driven environment as well as proven success developing customer loyaltyability to encourage a partner team and grow apple business
    
    Position for which resumes are shortlisted: Apple Solutions Consultant
    
    Shortlisted resume File names are as below
    
    53    24799301.pdf
    89    23568641.pdf
    23    20628003.pdf
    57    39718499.pdf
    56    25857360.pdf
    Name: File_name, dtype: object
    



```python
for each in df.to_list():
    print('Category: '+resume_data[['Category','File_name']][resume_data['File_name']==each].values[0][0]+'\n')
    print('File Name: '+resume_data[['Category','File_name']][resume_data['File_name']==each].values[0][1]+'\n')
    print('Resume Content: \n\n'+resume_data['resume_text'][resume_data['File_name']==each].values[0]+'\n')
    break
```

    Category: ACCOUNTANT
    
    File Name: 24799301.pdf
    
    Resume Content: 
    
    ACCOUNTANT
    Summary
    Accountant for a Medium sized Company
    Experience
    01/2009
     
    to 
    Current
    Accountant
     
    Company Name
     
    ï¼​ 
    City
     
    , 
    State
    Hired by their CPA firm to handle all accounting and job cost Reporting.
    01/2007
     
    to 
    01/2009
    Accountant
     
    Company Name
     
    ï¼​ 
    City
     
    , 
    State
    Hired by their CPA firm to handle all accounting functions..
    01/1997
     
    to 
    01/2007
    Accountant
     
    Company Name
     
    ï¼​ 
    City
     
    , 
    State
    Installed new Peachtree Accounting System.
    Installed new computer system using a local area network and Added a Web site.
    Education and Training
    1974
    B.S
     
    : 
    Business Administration Accounting
     
    University of Cincinnati
     
    ï¼​ 
    City
     
    , 
    State
     
    Business Administration Accounting
    Interests
    Annapolis Amblers Walking Club, President &Trailmaster, Maryland Volkssport Assn, President, Chesapeake Civil War Roundtable.
    Skills
    accounting, CPA, local area network, Peachtree Accounting, Reporting, Web site
    Additional Information
    Interests 
    Annapolis Amblers Walking Club, President &Trailmaster, Maryland Volkssport Assn, President, Chesapeake Civil War
    Roundtable.
    


