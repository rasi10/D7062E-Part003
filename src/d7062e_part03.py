import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ENSEMBLE METHODS - https://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator
# Bagging
from sklearn.ensemble import BaggingClassifier
# Extremely randomized trees
from sklearn.ensemble import ExtraTreesClassifier
# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
# Gradient tree boosting
from sklearn.ensemble import GradientBoostingClassifier
# The voting classifier
from sklearn.ensemble import VotingClassifier



def get_train_features_and_labels(input_file):
    gesture_training_data = pd.read_csv(input_file)
    train_features = gesture_training_data.iloc[:, 0:240].values
    train_labels = gesture_training_data.iloc[:, 241].values
    return [train_features, train_labels]


def handle_missing_values(features):
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(features)
    features = imputer.transform(features)
    return features


def perform_normalization_by_scaling(train, test):
    feature_scaler = StandardScaler()
    normalized_train = feature_scaler.fit_transform(train)
    normalized_test = feature_scaler.fit_transform(test)
    return [normalized_train, normalized_test]

def train_bagging_and_calculate_performance(train_features, train_labels, test_features, test_labels):
    print('######################################### Bagging ####################################################')
    
    # Train the Bagging Classifier
    bagging_clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5) 
    bagging_clf.fit(train_features, train_labels)

    # Make predictions
    predictions = bagging_clf.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame({'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        bagging_clf,
        train_features,
        train_labels,
        cv=10)
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')

def train_extremely_randomized_trees_and_calculate_performance(train_features, train_labels, test_features, test_labels):
    print('################################# EXTREMELY RANDOMIZED TREES ##########################################')
    
    # Train the Bagging Classifier
    extra_trees_classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    extra_trees_classifier.fit(train_features, train_labels)

    # Make predictions
    predictions = extra_trees_classifier.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame({'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        extra_trees_classifier,
        train_features,
        train_labels,
        cv=10)
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')


def train_adaboost_and_calculate_performance(train_features, train_labels, test_features, test_labels):
    print('######################################### AdaBoost ####################################################')
    
    # Train the Ada Boost Classifier
    ada_boost_clf = AdaBoostClassifier(n_estimators=100)    
    ada_boost_clf.fit(train_features, train_labels)

    # Make predictions
    predictions = ada_boost_clf.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame({'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        ada_boost_clf,
        train_features,
        train_labels,
        cv=10)
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions, zero_division=1))
    print('------------------------------------------------------------------------------------------------------')


def train_gradient_tree_boosting_and_calculate_performance(train_features, train_labels, test_features, test_labels):
    print('#################################### Gradient Tree Boosting ##########################################')
    
    # Train the Gradient Tree Boosting
    gradient_tree_boosting = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    gradient_tree_boosting.fit(train_features, train_labels)
    

    # Make predictions
    predictions = gradient_tree_boosting.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame({'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        gradient_tree_boosting,
        train_features,
        train_labels,
        cv=10)
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions, zero_division=1))
    print('------------------------------------------------------------------------------------------------------')

def train_voting_classifier_and_calculate_performance(train_features, train_labels, test_features, test_labels):
    print('##################################### THE VOTING CLASSIFIER ###########################################')
    
    
    logistic_regression_classifier = LogisticRegression(random_state=1, solver='lbfgs', max_iter=1000) 
    random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=1)
    gaussian_naive_bayes_classifier = GaussianNB()

    # Train the Gradient Tree Boosting
    voting_ensemble_classifier = VotingClassifier(estimators=[('lr', logistic_regression_classifier), ('rf', random_forest_classifier), ('gnb', gaussian_naive_bayes_classifier)], voting='hard') #eclf
    voting_ensemble_classifier.fit(train_features, train_labels)
    
    # Make predictions
    predictions = voting_ensemble_classifier.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame({'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')

    for clf, label in zip([logistic_regression_classifier, random_forest_classifier, gaussian_naive_bayes_classifier, voting_ensemble_classifier], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
        cross_v = cross_val_score(clf, train_features, train_labels, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (cross_v.mean(), cross_v.std(), label))
    
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')

if __name__ == "__main__":
    # Load datasets for training and test and share features and labels
    INPUT_FILE_TRAIN = 'datasets/train-final.csv'
    INPUT_FILE_TEST = 'datasets/test-final.csv'
    train_list = get_train_features_and_labels(INPUT_FILE_TRAIN)
    test_list = get_train_features_and_labels(INPUT_FILE_TEST)

    # Get features and labels out of the training and test datasets
    train_features = train_list[0]
    train_labels = train_list[1]
    test_features = test_list[0]
    test_labels = test_list[1]

    # Handle missing values in the training and test dataset
    train_features = handle_missing_values(train_features)
    test_features = handle_missing_values(test_features)

    # Normalizing the data
    normalized_data = perform_normalization_by_scaling(train_features, test_features)
    train_features = normalized_data[0]
    test_features = normalized_data[1]

    
    # Run performance evaluation of AdaBoost
    train_adaboost_and_calculate_performance(train_features, train_labels, test_features, test_labels)
    
    # Run performance evaluation of Extremely Randomized Trees
    train_extremely_randomized_trees_and_calculate_performance(train_features, train_labels, test_features, test_labels)
    
    # Run performance evaluation of AdaBoost
    train_bagging_and_calculate_performance(train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of Gradient Tree Boosting
    train_gradient_tree_boosting_and_calculate_performance(train_features, train_labels, test_features, test_labels)
    
    # The voting classifier
    train_voting_classifier_and_calculate_performance(train_features, train_labels, test_features, test_labels)