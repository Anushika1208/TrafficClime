
# 🚦 TrafficClime  

**Advanced Traffic Prediction Model using Deep Learning**  

## 📌 Problem Statement  
Urban areas worldwide face **severe traffic congestion**, leading to:  
- Increased travel times  
- Higher fuel consumption  
- Work delays  

**Weather conditions** (rain, snow, fog, extreme temperatures) further worsen traffic by causing:  
- **Accidents**  
- **Reduced road capacity**  
- **Slower traffic flow**  

To address these issues, a **reliable predictive system** is needed to forecast traffic disruptions due to adverse weather conditions.  

## 💡 Proposed Solution  
TrafficClime is an **AI-driven traffic forecasting model** that integrates **historical weather and traffic data** to predict future traffic conditions. Unlike traditional solutions, our model utilizes **deep learning algorithms** to capture complex patterns and deliver highly accurate traffic predictions.  

### ✅ Benefits:  
✔ **Helps city planners** optimize road management  
✔ **Supports traffic authorities** in decision-making  
✔ **Enables commuters** to avoid congestion and plan efficient travel routes  

## 🔥 Unique Features  
🔹 **Integration of Multiple Data Sources**:  
- Combines **historical weather, traffic, and event data** for accurate predictions.  

🔹 **Advanced Deep Learning Techniques**:  
- Leverages deep learning models to enhance prediction accuracy over time.  

🔹 **Event Impact Prediction**:  
- Analyzes past event data to estimate **how events impact traffic flow**.  

## 🛠️ Tools & Technologies Used  
### **Programming & Libraries**  
- **Python**  
- **TensorFlow, Keras** *(Deep Learning Frameworks)*  
- **Scikit-learn** *(Machine Learning Library)*  
- **Matplotlib, Seaborn** *(Data Visualization)*  

### **Development Environment**  
- **Google Colab** / **Visual Studio Code**  

## 🖥️ Hardware Requirements  
- **Processor:** Multi-core (Intel i5/i7 or AMD equivalent)  
- **GPU:** Dedicated GPU for deep learning training  
- **RAM:** Minimum **8GB**  

To **run the TrafficClime model**, follow these steps:  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/Anubhavjaiswal21/TrafficClime.git
cd TrafficClime
```  

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```  
(*If `requirements.txt` is missing, manually install TensorFlow, Pandas, NumPy, Scikit-learn, Matplotlib, and Seaborn.*)  

### **3️⃣ Preprocess the Dataset**  
Run `datapreprocessing.py` to preprocess the dataset:  
```bash
python datapreprocessing.py
```  
This will generate `train_preprocessed.csv`, `val_preprocessed.csv`, and `test_preprocessed.csv`.  

### **4️⃣ Train the Model (If Not Using Pretrained)**  
If you want to **train the model from scratch**, execute:  
```bash
python trafficflow.py
```  
This will create `traffic_flow_lstm_model.h5`.  

### **5️⃣ Predict Traffic Flow**  
Run the prediction script using the trained model:  
```bash
python traffic_predict_app.py
```  

### **6️⃣ View Visualizations**  
Execute:  
```bash
python visualizations.py
```  
(*If `visualizations.py` does not exist, check the README or generate plots manually using Matplotlib.*)  

## 📬 Contact  
📧 Email: [anushikasingh.2123@gmail.com]  
🔗 LinkedIn: [www.linkedin.com/in/anushika-singh-05260b237]  

---
