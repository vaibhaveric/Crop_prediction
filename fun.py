#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import *
#Importing Necessary Libraries
from load_model import model
import pandas as pd
from time import strptime
import numpy as np


# In[2]:

# Initializing the Flask
app = Flask(__name__)
mod,dic,ac = model()
# In[ ]:

#Main get Function
@app.route("/",methods=["GET"])
def main():
    return render_template("index.html",ans="",acc=int(ac*100))
@app.route("/predict",methods=['POST'])
def pred():
    #Accepting Only Post Request
    if request.method=='POST':
        crop = request.form['crop']
        wght = request.form['weight']
        year = request.form['year']
        mnth = request.form['month']
        mnth = strptime(mnth,'%b').tm_mon
        a = "".join([str(mnth),year])
        b = "010"+a
        test = []
        test.append(dic[str(crop)])
        test.append(float(wght))
        test.append(float(b))
        test = np.asarray(test)
        pred = mod.predict([test])
        for i in pred:
            s = int(ac*100)
            return render_template("index.html",ans=str(i),acc=s)
    else:
        return render_template("index.html",ans="",acc=int(ac*100))


if __name__=="__main__":
    app.run()
    
