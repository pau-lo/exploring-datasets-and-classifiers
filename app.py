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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


def main():
    st.title("Streamlit Example")

    st.write("""
    # Explore different classifiers and different datasets
    Which one works best?
    """)

    data_name = st.sidebar.selectbox("Select Dataset:",
                                     ("Breast Cancer", "Iris", "Wine",
                                      "Digits")
                                     )
    classifier_name = st.sidebar.selectbox("Select Classifier:",
                                           ("Logistic Regression", "KNN",
                                            "SVM", "Random Forest")
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
        # split data
        X = data.data
        y = data.target
        return X, y

    X, y = load_data()
    st.write("Load data: ", data_name)
    st.write("Shape of dataset", X.shape)

    def add_parameter(classifier_name):
        params = dict()
        if classifier_name == "Logistic Regression":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        elif classifier_name == "KNN":
            K = st.sidebar.slider("K", 1, 15)
            params["K"] = K
        elif classifier_name == "SVM":
            C = st.sidebar.slider("C", 0.01, 10.0)
            params["C"] = C
        else:
            # classifier_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth", 2, 15)
            n_estimators = st.sidebar.slider("n_estimators", 1, 100)
            params["max_depth"] = max_depth
            params["n_estimators"] = n_estimators
        return params

    params = add_parameter(classifier_name)

    def get_classifier(clf_name, params):
        if classifier_name == "Logistic Regression":
            clf = LogisticRegression(C=params["C"], solver='liblinear')
        elif classifier_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=params["K"])
        elif classifier_name == "SVM":
            clf = SVC(C=params["C"])
        else:
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"]
            )
        return clf

    clf = get_classifier(classifier_name, params)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23)

    scaler = MinMaxScaler()
    scaler.fit_transform(X_train)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    accuracy_results = {
        "model_name": classifier_name,
        "model_accuracy": accuracy,
    }

    st.write("Model Results: ")
    st.json(accuracy_results)

    pca = PCA(n_components=2, svd_solver='full')
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    st.write(f"Principal Component Analysis for {data_name}")

    def pca_plot():
        fig = plt.figure(1, figsize=(8, 6))
        # plt.clf()
        # ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='Paired')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar()
        st.pyplot(fig)
    pca_plot()


if __name__ == "__main__":
    main()
