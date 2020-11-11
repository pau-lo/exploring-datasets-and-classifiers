import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    plot_confusion_matrix
)
from sklearn.decomposition import PCA


def main():
    st.title("A webapp that explores different datasets and classifiers.")

    st.write("""
    ### Explore the Breast Cancer, Iris, Wine and Digits datasets using Scikit Learn.
    """)

    st.sidebar.write("### Choose the best classifier and parameter")
    data_name = st.sidebar.selectbox("Select Dataset:",
                                     ("Breast Cancer", "Iris",
                                      "Wine", "Digits")
                                     )
    classifier_name = st.sidebar.selectbox("Select Classifier:",
                                           ("Logistic Regression",
                                            "K-Nearest Neighbors (KNN)",
                                            "Support Vector Machine (SVM)",
                                            "Random Forest")
                                           )

    @st.cache(persist=True)
    def load_data():
        if data_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        elif data_name == "Iris":
            data = datasets.load_iris()
        elif data_name == "Wine":
            data = datasets.load_wine()
        else:
            data = datasets.load_digits()

        # split features and target
        X = data.data
        y = data.target
        class_names = data.target_names
        return X, y, class_names

    X, y, class_names = load_data()

    st.write("Current data: ", data_name)
    st.write("Shape of", data_name, "dataset ", X.shape)
    st.write("Class labels: ",  class_names)

    st.sidebar.write("Adjust parameters: ")

    def add_parameter(classifier_name):
        params = dict()
        if classifier_name == "Logistic Regression":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        elif classifier_name == "K-Nearest Neighbors (KNN)":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif classifier_name == "Support Vector Machine (SVM)":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        else:
            # load random forest
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        return params

    params = add_parameter(classifier_name)

    def get_classifier(clf_name, params):
        if classifier_name == "Logistic Regression":
            clf = LogisticRegression(C=params["C"], solver='liblinear')
        elif classifier_name == "K-Nearest Neighbors (KNN)":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif classifier_name == "Support Vector Machine (SVM)":
            clf = SVC(C=params["C"])
        else:
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"]
            )
        return clf

    clf = get_classifier(classifier_name, params)

    # split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23)

    # standardized features
    scaler = StandardScaler()
    scaler.fit_transform(X_train)

    # fit model
    clf.fit(X_train, y_train)

    # predict
    y_pred = clf.predict(X_test)

    # calculate metrics
    accuracy = accuracy_score(y_test, y_pred).round(3)
    precision = precision_score(y_test, y_pred,
                                average='weighted', zero_division=0).round(3)
    recall = recall_score(y_test, y_pred,
                          average='weighted').round(3)

    model_results = {
        "Model Name": classifier_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall Score": recall
        # "ROC AUC Score": area_under_curve
    }

    st.write("Model Results: ")
    st.json(model_results)

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # confusiont matrix plot
    def confusion_matrix_plot():
        st.subheader("Confusion Matrix")
        plt.figure(1, figsize=(9, 7))
        plot_confusion_matrix(clf, X_test, y_test,
                              display_labels=class_names)
        st.pyplot()
        
    # instantiate plot
    confusion_matrix_plot()

    # pca plot

    pca = PCA(n_components=2, svd_solver='full')
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    # st.write(f"Principal Component Analysis for {data_name}")

    def pca_plot():
        st.subheader("PCA Scatter Plot")
        fig = plt.figure(1, figsize=(9, 7))
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='Paired')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()
        st.pyplot(fig)

    # instantiate plot
    pca_plot()


if __name__ == "__main__":
    main()
