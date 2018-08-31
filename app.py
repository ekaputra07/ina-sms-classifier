import os
from flask import Flask, request, jsonify
from sklearn.multiclass import *
from sklearn.svm import *
import pandas
import joblib
import pickle

application = Flask(__name__)
global Vectorizer
global Classifier

SUCCESS = "success"
ERROR = "error"

def init_model_clasifier(application):
  # load vectorizer
  global Vectorizer
  Vectorizer = joblib.load('model/vectorizer.pkl')

  # load clasifiers
  objects = []
  with open('model/model', 'rb') as f:
    while True:
      try:
        objects.append(pickle.load(f))
      except EOFError:
        break

  global Classifier
  Classifier = objects[0]

init_model_clasifier(application)

def predict(text):
  input_to_classifier = [text]

  try:
    vectorize_input = Vectorizer.transform(input_to_classifier)
    output = Classifier.predict(vectorize_input)
    predict = output[0]
    return SUCCESS, {"message": text, "type": predict}

  except BaseException as inst:
    error = str(type(inst).__name__) + ' ' + str(inst)
    return ERROR, error

@application.route('/classify/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    if len(message) > 0:
      status, result = predict(message)
      if status == SUCCESS:
        return jsonify(status=status, result=result), 200
      else:
        return jsonify(status=ERROR, error=result), 500
    else:
      return jsonify(status=ERROR, error="no input specified"), 500


if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8000, debug=True, use_reloader=True)
