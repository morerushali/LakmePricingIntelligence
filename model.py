import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Excel Path
file_path = r"C:/Users/rushu/OneDrive/Desktop/LAKME PRICE INTELLIGENCE/Lakme/Lakmepart2.xlsx"

data = pd.read_excel(file_path)

X = data[["ProductID","MRP","Discount","GST"]]
y = data["FinalPrice"]

model = LinearRegression()
model.fit(X,y)

joblib.dump(model,"price_model.pkl")

print("✅ Model trained successfully")