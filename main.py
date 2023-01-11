from flask import Flask, request
from data_utils import preprocess_data, preprocess_predict_data
import threading

# Preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

data_lock = threading.Lock()
x = []
y = []

training_thread = None

# Model and friends
dictVectorizer = DictVectorizer(sparse=False)
labelEncoder = LabelEncoder()
clf = RandomForestClassifier(max_depth=2, random_state=0)

def train_model():
    with data_lock:
        X = dictVectorizer.fit_transform(x)
        Y = labelEncoder.fit_transform(y)
    clf.fit(X, Y)
    print('Finished training model')

@app.route('/feed', methods=['POST'])
def feed():
    global training_thread

    if request.method != 'POST':
        return 'bad request', 400

    nx, ny = preprocess_data(request.json)

    with data_lock:
        x.extend(nx)
        y.extend(ny)

    training_thread = threading.Thread(target=train_model)
    training_thread.start()

    return 'success', 200

@app.route('/predict', methods=['POST'])
def predict():
    global training_thread

    if training_thread != None:
        training_thread.join()
        training_thread = None

    try:
        x = preprocess_predict_data(request.json)
        X = dictVectorizer.transform(x)
        predicted = clf.predict(X)
        inv_label = labelEncoder.inverse_transform(predicted)
        if inv_label != None:
            return str(inv_label[0]), 200
        else:
            return '', 200
    except:
        return '', 200


@app.route('/')
def index():
    return 'PR mini-project web-server running'

app.run(host='0.0.0.0', port=8000)
