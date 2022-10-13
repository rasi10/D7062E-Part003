import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


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


def train_decision_tree_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('###################################### DECISION TREE ALGORITHM ######################################')

    # Train the decision tree algorithm
    dt_reg = DecisionTreeClassifier()
    dt_reg.fit(train_features, train_labels)

    # Make predictions
    predictions = dt_reg.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        dt_reg,
        train_features,
        train_labels,
        cv=10,
        scoring="accuracy")
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')


def train_SVM_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('########################################### SVM ALGORITHM ###########################################')

    # Train the random forest algorithm
    svm_reg = svm.SVC(decision_function_shape='ovo')
    svm_reg.fit(train_features, train_labels)

    # Make predictions
    predictions = svm_reg.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        svm_reg,
        train_features,
        train_labels,
        cv=10,
        scoring="accuracy")
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')


def train_KNN_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('########################################### KNN ALGORITHM ###########################################')
    # Train the random forest algorithm
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(train_features, train_labels)

    # Make predictions
    predictions = knn_clf.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        knn_clf,
        train_features,
        train_labels,
        cv=10,
        scoring="accuracy")
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')


def train_MLP_and_calculate_performance(
        train_features,
        train_labels,
        test_features,
        test_labels):
    print('########################################### MLP ALGORITHM ###########################################')

    # Train the random forest algorithm
    mlp_clf = MLPClassifier(random_state=1, max_iter=300)
    mlp_clf.fit(train_features, train_labels)

    # Make predictions
    predictions = mlp_clf.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        mlp_clf,
        train_features,
        train_labels,
        cv=10,
        scoring="accuracy")
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions))
    print('------------------------------------------------------------------------------------------------------')


def train_random_forest_and_calculate_performance(
        train_features, train_labels, test_features, test_labels):
    print('###################################### RANDOM FOREST ALGORITHM ######################################')

    # Train the random forest algorithm
    rf_reg = RandomForestClassifier(max_depth=2, random_state=0)
    rf_reg.fit(train_features, train_labels)

    # Make predictions
    predictions = rf_reg.predict(test_features)

    # Run a comparison
    comparison = pd.DataFrame(
        {'Real': test_labels, 'Predictions': predictions})
    # print(comparison)

    # print 10-fold cross validation score
    print('------------------------------------------------------------------------------------------------------')
    print('Cross validation with 10-fold')
    cross_v = cross_val_score(
        rf_reg,
        train_features,
        train_labels,
        cv=10,
        scoring="accuracy")
    print(cross_v)
    print(
        f'Average accuracy out of the cross-validation with 10 folds: {cross_v.mean()}')
    print('------------------------------------------------------------------------------------------------------')

    # Print classification report
    # print(confusion_matrix(test_labels, predictions))
    # print(f'Accuracy Score {np.sqrt(accuracy_score(test_labels, predictions))}')
    print(classification_report(test_labels, predictions, zero_division=1))
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
    normalized_data = perform_normalization_by_scaling(
        train_features, test_features)
    train_features = normalized_data[0]
    test_features = normalized_data[1]

    # Run performance evaluation of Decision Tree
    train_decision_tree_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of SVM
    train_SVM_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of KNN
    train_KNN_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of MLP
    train_MLP_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)

    # Run performance evaluation of Random Forest
    train_random_forest_and_calculate_performance(
        train_features, train_labels, test_features, test_labels)
