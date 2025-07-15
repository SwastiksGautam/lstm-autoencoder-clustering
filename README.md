# Deep Learning: Time Series Forecasting & Autoencoder Clustering

This repository showcases two deep learning applications:
1. 📈 LSTM-based Time Series Forecasting
2. 🔍 Autoencoder-based Dimensionality Reduction + KMeans Clustering on MNIST

---

## 📁 1. LSTM Time Series Forecasting

- Generates synthetic sinusoidal time series data.
- Uses LSTM to learn patterns and predict future values.
- Includes loss curve, forecast vs actual, and residual analysis.

📌 Key Features:
- Synthetic data with noise
- Sequential LSTM model
- EarlyStopping & ReduceLROnPlateau
- Visualization: Forecasting + Residuals

---

## 📁 2. Autoencoder + KMeans Clustering on MNIST

- Trains a deep autoencoder to compress MNIST images into 32D latent space.
- Applies KMeans on latent space for clustering.
- Uses t-SNE for 2D visualization of digit separation.

📌 Key Features:
- Fully connected Autoencoder
- KMeans Clustering on Latent Space
- t-SNE visualization
- True labels vs predicted cluster visualization
- Silhouette Score for cluster quality

---

## 📦 Dependencies

```bash
pip install numpy matplotlib scikit-learn tensorflow
