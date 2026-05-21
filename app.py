import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from haversine import haversine, Unit

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam

st.title("Food Delivery Time Prediction & Clustering")

uploaded_file = st.file_uploader("Upload Food_Delivery_Time_Prediction.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    # preprocessing
    df.fillna(df.mean(numeric_only=True), inplace=True)

    median_time = df['Delivery_Time'].median()
    df['Target'] = (df['Delivery_Time'] > median_time).astype(int)

    df[['Lat1','Lon1']] = df['Customer_Location'].str.strip('()').str.split(',', expand=True).astype(float)
    df[['Lat2','Lon2']] = df['Restaurant_Location'].str.strip('()').str.split(',', expand=True).astype(float)

    df['Distance_km'] = df.apply(
        lambda row: haversine(
            (row['Lat1'], row['Lon1']),
            (row['Lat2'], row['Lon2']),
            unit=Unit.KILOMETERS
        ),
        axis=1
    )

    scaler = MinMaxScaler()
    df[['Distance', 'Delivery_Time']] = scaler.fit_transform(df[['Distance', 'Delivery_Time']])

    df = pd.get_dummies(df, columns=[
        'Weather_Conditions',
        'Traffic_Conditions',
        'Order_Priority',
        'Order_Time',
        'Vehicle_Type'
    ])

    X_cluster = df.drop([
        'Order_ID',
        'Delivery_Time',
        'Customer_Location',
        'Restaurant_Location'
    ], axis=1)

    scaler2 = StandardScaler()
    X_scaled = scaler2.fit_transform(X_cluster)

    st.header("Clustering")

    # elbow method
    wcss = []
    for i in range(1, 8):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 8), wcss, marker='o')
    ax1.set_title("Elbow Method")
    st.pyplot(fig1)

    # kmeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Distance'], df['Delivery_Time'], c=df['Cluster_KMeans'])
    ax2.set_title("KMeans Clustering")
    st.pyplot(fig2)

    # dendrogram
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sch.dendrogram(sch.linkage(X_scaled, method='ward'))
    st.pyplot(fig3)

    # hierarchical
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    df['Cluster_HC'] = hc.fit_predict(X_scaled)

    fig4, ax4 = plt.subplots()
    ax4.scatter(df['Distance'], df['Delivery_Time'], c=df['Cluster_HC'])
    ax4.set_title("Hierarchical Clustering")
    st.pyplot(fig4)

    st.header("Classification")

    X = df.drop([
        'Order_ID',
        'Delivery_Time',
        'Target',
        'Customer_Location',
        'Restaurant_Location'
    ], axis=1)

    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler3 = StandardScaler()
    X_train = scaler3.fit_transform(X_train)
    X_test = scaler3.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    st.subheader("Logistic Regression Results")
    st.write("Accuracy:", accuracy_score(y_test, y_pred_lr))
    st.write("Precision:", precision_score(y_test, y_pred_lr))
    st.write("Recall:", recall_score(y_test, y_pred_lr))
    st.write("F1 Score:", f1_score(y_test, y_pred_lr))

    st.header("Neural Network Models")

    # # Neural Network 1
    # model = Sequential([
    #     Dense(64, activation='relu', input_dim=X_train.shape[1]),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model.compile(
    #     optimizer=Adam(learning_rate=0.001),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    # with st.spinner("Training Neural Network 1..."):
    #     model.fit(
    #         X_train, y_train,
    #         epochs=30,
    #         batch_size=16,
    #         validation_split=0.2,
    #         verbose=0
    #     )

    # y_pred_nn = (model.predict(X_test) > 0.5).astype(int)

    # st.subheader("Neural Network 1 Results")
    # st.write("Accuracy:", accuracy_score(y_test, y_pred_nn))
    # st.write("Precision:", precision_score(y_test, y_pred_nn))
    # st.write("Recall:", recall_score(y_test, y_pred_nn))
    # st.write("F1 Score:", f1_score(y_test, y_pred_nn))

    # # Neural Network 2
    # model2 = Sequential([
    #     Dense(128, activation='relu', input_dim=X_train.shape[1]),
    #     Dense(64, activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # model2.compile(
    #     optimizer=Adam(learning_rate=0.0005),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )

    # with st.spinner("Training Neural Network 2..."):
    #     model2.fit(
    #         X_train, y_train,
    #         epochs=50,
    #         batch_size=8,
    #         validation_split=0.2,
    #         verbose=0
    #     )

    # y_pred_nn2 = (model2.predict(X_test) > 0.5).astype(int)

    # st.subheader("Neural Network 2 Results")
    # st.write("Accuracy:", accuracy_score(y_test, y_pred_nn2))
    # st.write("Precision:", precision_score(y_test, y_pred_nn2))
    # st.write("Recall:", recall_score(y_test, y_pred_nn2))
    # st.write("F1 Score:", f1_score(y_test, y_pred_nn2))