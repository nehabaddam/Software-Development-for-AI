import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from src.utils.timer import fit_with_time, predict_with_time
from tests.tree.fixtures import dummy_labels_t5, dummy_titanic_t5, dummy_titanic_lr


def test_train_t5(dummy_titanic_lr, dummy_titanic_t5):
    
    model = dummy_titanic_lr
    
    X_train, y_train,X_test,y_test = dummy_titanic_t5

    predictions = model.predict(X_train)
    predicted_train = np.round(predictions)

    accuracy_train = accuracy_score(y_train, predicted_train)
    assert accuracy_train > 0.60

    predictions_test = model.predict(X_test)
    predicted_test = np.round(predictions_test)

    accuracy_test = accuracy_score(y_test, predicted_test)
    assert accuracy_test > 0.60


def test_logreg_serving_latency(dummy_titanic_t5):

    X_train, y_train, X_test, y_test = dummy_titanic_t5

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    latency = np.array([predict_with_time(lr, X_test)[1] for i in range(200)])
    serving_latency = np.quantile(latency, 0.99)

    assert serving_latency < 0.005, 'Serving latency at 99th percentile should be less than 0.005 sec'

def test_logreg_training_time_t5(dummy_titanic_t5):

    X_train, y_train, X_test, y_test = dummy_titanic_t5

    lr = LogisticRegression(max_iter=1000)

    latency = np.array([fit_with_time(lr, X_train, y_train)[1] for i in range(50)])
    time = np.quantile(latency, 0.95)

    assert time < 1.0, 'Training time at 95th percentile should be less than 1.0 sec'


def test_increase_acc(dummy_titanic_t5):

    X_train, y_train, _, _ = dummy_titanic_t5

    accuracy_list= []
    for iterations in range(1000, 1010):
        lr = LogisticRegression(max_iter= iterations)
        lr.fit(X_train, y_train)
        predictions = lr.predict(X_train)
        predict_binary = np.round(predictions)
        accuracy_list.append(accuracy_score(y_train, predict_binary))

    assert sorted(accuracy_list) == accuracy_list, 'Accuracy should increase as the number of iterations increases.'


def test_lr_overfit(dummy_labels_t5, dummy_titanic_t5):

    feats, labels = dummy_labels_t5

    lr =LogisticRegression(max_iter=1000)
    lr.fit(feats, labels)

    predictions = np.round(lr.predict(feats))

    assert np.array_equal(labels, predictions), 'Logistic Regression should fit data perfectly and prediction should == labels.'

    X_train, y_train, X_test, y_test = dummy_titanic_t5

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    predict_train = lr.predict(X_train)
    predicted_train_binary = np.round(predict_train)

    accuracy_train = accuracy_score(y_train, predicted_train_binary)

    assert accuracy_train > 0.75, 'Accuracy on training data should be greater than 0.75'
    
