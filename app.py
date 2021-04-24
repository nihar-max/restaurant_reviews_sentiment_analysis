# Importing essential libraries
from flask import Flask, render_template, request
import pickle

# Load the Logistic model and TF-IDF object from disk
filename = 'restaurant-sentiment-logistic-model.pkl'
clf_log = pickle.load(open(filename, 'rb'))
tf_idf = pickle.load(open('tf-transform.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = tf_idf.transform(data).toarray()
    	my_prediction = clf_log.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)