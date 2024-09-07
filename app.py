from flask import Flask, render_template, request, url_for, session, redirect, flash
import re
import pickle
import pandas as pd
import pandas as pd
import re 
from flask import Flask, redirect, url_for, request, render_template

from tensorflow.keras.models import load_model
import pickle
import h5py
import numpy as np # linear algebra
import pandas as pd  
import tensorflow as tf


from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__) #Initialize the flask App

MODEL_PATH ='lstms.h5'
print(MODEL_PATH)
model = load_model(MODEL_PATH)
 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
@app.route('/')

@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/chart')
def chart():
	return render_template('chart.html')

#@app.route('/future')
#def future():
#	return render_template('future.html')    

@app.route('/login')
def login():
	return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	


#@app.route('/home')
#def home():
 #   return render_template('home.html')

@app.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    return render_template('prediction.html')


#@app.route('/upload')
#def upload_file():
#   return render_template('BatchPredict.html')



@app.route('/predict',methods=['POST','GET'])
def predict():
  
    if request.method == 'POST':    
        
        query_content=request.form['news_content']
        
        total= query_content
        total = re.sub('<[^>]*>', '', total)
        total = re.sub(r'[^\w\s]','', total)
        total = total.lower()     
        data=[total]
        twt = tokenizer.texts_to_sequences(data)
        twt = pad_sequences(twt, maxlen=30, dtype='int32', value=0)
        # transform data
        sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
        if(np.argmax(sentiment) == 0):
              pred= "No_bullying"
        elif (np.argmax(sentiment) == 1):
              pred="bullying"
        print(pred)      
       
    return render_template('prediction.html', prediction_text= pred) 
 
@app.route('/performance')
def performance():
	return render_template('performance.html')   
    
if __name__ == "__main__":
    app.run(debug=True)
