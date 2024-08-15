from flask import Flask , request , jsonify # request to handle form data 
import pickle
import numpy as np 
import sklearn
print(sklearn.__version__)


#  load pickle data 
model = pickle.load(open('model_placement.pk1','rb'))
app =Flask(__name__)

@app.route('/')
def home():
    return "hellow world hiii how r u "

#  api exist at this path 
@app.route('/predict',methods = ['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    porfile_score = request.form.get('porfile_score')
    # Debugging output
    print(f"Received - CGPA: {cgpa}, IQ: {iq}, Profile Score: {porfile_score}")

    input_querry = np.array([[cgpa,iq,porfile_score]])
    model.predict(input_querry)[0]
    #  return these requests in json 

    result  = {'cgpa' : cgpa, 'iq' : iq , 'porfile_score':porfile_score}

    return jsonify({'placement':str(result)})
       #   api completed 

if __name__ == '__main__':
    app.run(debug = True)

#  one error 
# AttributeError: 'DecisionTreeClassifier' object has no attribute 'monotonic_cst'
