import numpy as np
from flask import Flask, request, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('home.html', prediction_text=" {} ° F".format(round(prediction[0]),5))

if __name__ == '__main__':
    app.run(debug=True)