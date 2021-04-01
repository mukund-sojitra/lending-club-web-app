import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.model import load_model

app = Flask(__name__)

# model = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
# model = pickle.load(open('logisticregression.pkl', 'rb'))

model = tensorflow.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = np.array(int_features).reshape(1,4)
    
    # print(final_features)
    # print(final_features.shape)

    # prediction = model.predict(final_features)

    prediction = model.predict_classes(final_features)
    # print(prediction)

    if int(prediction) == 0:
        output = "No"
        return render_template('index.html', prediction_text='Sorry, Your loan request has been not approved by Lending Club!!')
    else:
        output = "Yes"
        return render_template('index.html', prediction_text='{}, Your loan request has been approved by Lending Club!!'.format(output))

    # output = round(prediction[0], 2)

    # return render_template('index.html', prediction_text='{}, Your loan request has been approved by Lending Club!!'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)