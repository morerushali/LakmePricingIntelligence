from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import os

app = Flask(__name__)

file_path = os.path.join(os.path.dirname(__file__), "Lakmepart2.xlsx")
data = pd.read_excel(file_path)


model = joblib.load(os.path.join(os.path.dirname(__file__), "price_model.pkl"))


# ================= HOME PAGE =================
@app.route("/")
def home():

    products = data.to_dict(orient="records")

    return render_template(
        "index.html",
        products=products
    )


# ================= AI PRICE PREDICTION =================
@app.route("/predict", methods=["POST"])
def predict():

    product_id = int(request.form["product_id"])

    row = data[data["ProductID"] == product_id].iloc[0]

    mrp = row["MRP"]
    discount = row["Discount"]
    gst = row["GST"]

    # AI Prediction
    predicted_price = model.predict([[product_id,mrp,discount,gst]])[0]

    final_price = round(predicted_price,2)

    savings = round(mrp - final_price,2)

    profit = round(final_price * 0.30,2)

    ai_discount = round(discount + 2,2)

    demand_score = round((100-discount)*0.6 + (mrp/100)*0.4,2)

    return render_template(
        "result.html",
        product=row["ProductName"],
        mrp=mrp,
        discount=discount,
        gst=gst,
        price=final_price,
        savings=savings,
        profit=profit,
        ai_discount=ai_discount,
        demand_score=demand_score,
        confidence="98%"
    )


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():

    labels = data["ProductName"].tolist()

    mrp = data["MRP"].tolist()

    final = (
        (data["MRP"]-(data["MRP"]*data["Discount"]/100))
        +((data["MRP"]-(data["MRP"]*data["Discount"]/100))
        *data["GST"]/100)
    ).round(2).tolist()

    profit = [round(p*0.30,2) for p in final]

    discount = data["Discount"].tolist()

    demand = [(100-d)*0.6+(m/100)*0.4 for d,m in zip(discount,mrp)]

    top_product = data.sort_values("MRP",ascending=False).iloc[0]["ProductName"]

    return render_template(
        "dashboard.html",
        labels=labels,
        mrp=mrp,
        final=final,
        profit=profit,
        discount=discount,
        demand=demand,
        top_product=top_product
    )


# ================= DOWNLOAD REPORT =================
@app.route("/download")
def download():

    data.to_csv("lakme_ai_pricing_report.csv",index=False)

    return send_file(
        "lakme_ai_pricing_report.csv",
        as_attachment=True
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)