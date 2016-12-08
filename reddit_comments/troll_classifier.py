import numpy as np
import sys
from sklearn import svm

import settings
import logging
logger = logging.getLogger(__name__)

    /*
    * parse_file is a wrapper. It loops through each line in
    * the filename passed to it, and sends it to the defined function
    * with the feature_words as a parameter.
    */
def parse_file(filename, func=None, feature_words=None):
    """
    Parses a file into data and returns a numpy array
    If a function is passed as func, then the function will be called on the
    the raw line of data to manipulate it as you want
    """
    data_file = open(filename, 'r')
    data_set = []
    feature_vectors = []
    label_vectors = []
    contents = []
    for line in data_file.readlines():
        if func:
            feature_vector, label, content  = func(line, feature_words)
            feature_vectors.append(feature_vector)
            label_vectors.append(label)
            contents.append(content)
        else:
            data_point = line.replace("\n", "")
            data_set.append(data_point)
    data_file.close()
    if len(data_set) != 0:
        return np.array(data_set)
    else:
        data_set.append(np.array(feature_vectors))
        data_set.append(np.array(label_vectors))
        data_set.append(contents)
        return data_set


def train_troll_classifier(words, training_data):
    linear_svc = svm.LinearSVC()
    linear_svc.fit_transform(training_data[0], training_data[1])
    return linear_svc

    /*
    * Loops through the feature_words, and sets up an array (vector).
    * The array contains a 1 or a 0 for each feature word. A 1 if the
    * comment contains the feature word, and a 0 if it doesn't. -BY
    */
def parse_insult_data_set_line(line, feature_words):
    feature_vector = [0 for x in xrange(len(feature_words))]
    cols = line.split(",")
    label = 1 if cols[0] == '1' else -1
    content = cols[2].replace('"""', '')
    list_of_words = content.split(" ")
    for index, feature in enumerate(feature_words):
        for word in list_of_words:
            if word == feature:
                feature_vector[index] = 1
    return np.array(feature_vector), label, content


def convert_content_to_vector(content, feature_words):
    feature_vector = [0 for x in xrange(len(feature_words))]
    list_of_words = content.split(" ")
    for index, feature in enumerate(feature_words):
        for word in list_of_words:
            if word.lower() == feature:
                feature_vector[index] = 1
    return np.array(feature_vector)


def test_troll_classifier(model, test_data):
    return model.predict(test_data[0])

    /*
    * it_is_a_troll appears to be the final classification function.
    * Once the model has been trained, new comments are passed to this
    * function to see where they fall on the troll-scale given the
    * training of the model.
    */
def it_is_a_troll(model, feature_words, content):
    feature_vector = convert_content_to_vector(content, feature_words)
    prediction = model.predict(feature_vector)
    return True if prediction == 1 else False

    /*
    * load_troll_classification_model appears to be the method
    * to train the model.
    */
def load_troll_classification_model():
    logger.info("TRAINING CLASSIFICATION MODEL")
    feature_words = parse_file(settings.TROLL_WORDS)
    training_data = parse_file(settings.TROLL_TRAIN,
                               func=parse_insult_data_set_line,
                               feature_words=feature_words)
    linear_model = train_troll_classifier(feature_words, training_data)
    logger.info("CLASSIFICATION MODEL IS TRAINED")
    return linear_model, feature_words


def main():
    words_file = sys.argv[1]
    training_data_file = sys.argv[2]
    test_data_file = sys.argv[3]
    feature_words = parse_file(words_file)
    training_data = parse_file(training_data_file, func=parse_insult_data_set_line, feature_words=feature_words)
    test_data = parse_file(test_data_file, func=parse_insult_data_set_line, feature_words=feature_words)
    print "finished parsing files"
    linear_model = train_troll_classifier(feature_words, training_data)
    predictions = test_troll_classifier(linear_model, test_data)
    print predictions
    for index, prediction in enumerate(predictions):
        if prediction == 1:
            print prediction, test_data[2][index]
    params = linear_model.get_params(deep=True)
    print params


if __name__ == '__main__':
    main()
