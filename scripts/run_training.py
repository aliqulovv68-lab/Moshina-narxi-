from src.models.train import load_data, preprocess, train_model, save_model

# Bu yerda src/ dagi funksiyalar chaqirilyapti - bu scripts papkasining vazifasi
df = load_data("Data/raw/used_car_price_prediction_1M.csv")
X, y = preprocess(df, target_col="Price")
model = train_model(X, y)
save_model(model)

print("Pipeline tugadi! src/ dagi funksiyalar scripts/ orqali ishga tushirildi.")
