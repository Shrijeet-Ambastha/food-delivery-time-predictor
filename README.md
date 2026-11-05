# Food Delivery Time Predictor

A Streamlit app that analyses food delivery orders and predicts whether an order will be **delivered faster or slower than the median delivery time**, based on factors such as distance, weather, traffic, order priority, order time and vehicle type. It also groups orders into clusters to reveal delivery patterns.

## Features

- Interactive Streamlit dashboard with a dataset preview
- Data preprocessing: missing-value imputation, feature scaling and one-hot encoding
- Distance calculation between customer and restaurant using the **Haversine formula**
- **Clustering** with K-Means (including an elbow-method plot) and Hierarchical clustering (with a dendrogram)
- **Classification** with Logistic Regression, reporting accuracy, precision, recall and F1 score
- Optional neural network experiments (included in `app.py` as commented-out code, requires TensorFlow)

## Dataset

`Food_Delivery_Time_Prediction.csv` contains 200 orders with the following columns:

| Column                       | Description                                  |
|------------------------------|----------------------------------------------|
| `Order_ID`                   | Unique order identifier                      |
| `Customer_Location`          | Customer coordinates (latitude, longitude)   |
| `Restaurant_Location`        | Restaurant coordinates (latitude, longitude) |
| `Distance`                   | Distance between customer and restaurant     |
| `Weather_Conditions`         | Weather at the time of delivery              |
| `Traffic_Conditions`         | Traffic level                                |
| `Delivery_Person_Experience` | Experience of the delivery person            |
| `Order_Priority`             | Priority of the order                        |
| `Order_Time`                 | Time of day of the order                     |
| `Vehicle_Type`               | Vehicle used for delivery                    |
| `Restaurant_Rating`          | Rating of the restaurant                     |
| `Customer_Rating`            | Rating given by the customer                 |
| `Delivery_Time`              | Delivery time (target basis)                 |
| `Order_Cost`                 | Cost of the order                            |
| `Tip_Amount`                 | Tip given                                    |

## How It Works

1. **Load and clean**: the CSV is read and missing numeric values are filled with column means.
2. **Create the target**: orders with a delivery time above the median are labelled `1` (slow), the rest `0` (fast).
3. **Engineer features**: location strings are parsed into coordinates and a Haversine distance in kilometres is computed.
4. **Scale and encode**: numeric features are scaled and categorical features (weather, traffic, priority, order time, vehicle) are one-hot encoded.
5. **Cluster**: K-Means (k = 3) and Agglomerative clustering (k = 3, Ward linkage) group similar orders, visualised with scatter plots.
6. **Classify**: the data is split 80/20 into train and test sets and a Logistic Regression model is trained and evaluated.

## Tech Stack

| Area              | Technology                     |
|-------------------|--------------------------------|
| Language          | Python                         |
| Web app           | Streamlit                      |
| Data handling     | pandas, NumPy                  |
| Visualisation     | Matplotlib                     |
| Machine learning  | scikit-learn, SciPy            |
| Geo distance      | haversine                      |

## Project Structure

```
food-delivery-time-predictor/
├── app.py                              # Streamlit application
├── Food_Delivery_Time_Prediction.csv   # Dataset
└── requirements.txt                    # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip

### Installation

```bash
git clone https://github.com/Shrijeet-Ambastha/food-delivery-time-predictor.git
cd food-delivery-time-predictor

# optional but recommended
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Enable the neural network models (optional)

The neural network sections in `app.py` are commented out. To use them, install TensorFlow, then uncomment the TensorFlow imports and the "Neural Network" blocks:

```bash
pip install tensorflow
```

## Output

When you run the app you will see:

- A preview of the dataset
- An elbow-method plot to choose the number of clusters
- K-Means and Hierarchical cluster scatter plots, plus a dendrogram
- Logistic Regression metrics: accuracy, precision, recall and F1 score

## Future Improvements

- Predict the exact delivery time with a regression model
- Add an input form so users can enter an order and get a prediction
- Compare more models (Random Forest, Gradient Boosting)
- Use a larger real-world dataset

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a pull request

## Author

**Shrijeet Ambastha**: [GitHub](https://github.com/Shrijeet-Ambastha)
