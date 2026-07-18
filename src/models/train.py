import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

from src.utils.logger import get_logger

logger = get_logger(
    name="trainer",
    log_file="log/trainer.log",
    error_file="log/errors/training_errors.log"
)


def load_data(path):
    try:
        logger.info(f"Data yuklanmoqda: {path}")
        df = pd.read_csv(path)
        logger.info(f"Data yuklandi: {df.shape[0]} qator, {df.shape[1]} ustun")
        return df
    except FileNotFoundError as e:
        logger.error(f"Fayl topilmadi: {e}")
        raise
    except Exception as e:
        logger.error(f"Data yuklashda xatolik: {e}")
        raise


def preprocess(df, target_col="Price"):
    try:
        logger.info("Preprocessing boshlandi")
        df = df.dropna()
        logger.info(f"NaN qatorlar olib tashlandi, qoldi: {df.shape[0]} qator")

        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        logger.info(f"Kategorik ustunlar encode qilindi: {list(cat_cols)}")

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Ustunlar scale qilindi")

        return X_scaled, y
    except KeyError as e:
        logger.error(f"Target ustun topilmadi: {e}")
        raise
    except Exception as e:
        logger.error(f"Preprocessingda xatolik: {e}")
        raise


def train_model(X, y):
    try:
        logger.info("Train/test bolish boshlandi")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        logger.info(f"Train: {X_train.shape[0]} qator, Test: {X_test.shape[0]} qator")

        logger.info("Model train qilinmoqda (XGBoost)")
        model = XGBRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model muvaffaqiyatli train qilindi")

        score = model.score(X_test, y_test)
        logger.info(f"Test R2 score: {score:.4f}")

        return model
    except MemoryError as e:
        logger.error(f"MemoryError: xotira yetmadi - {e}")
        raise
    except Exception as e:
        logger.error(f"Train qilishda xatolik: {e}")
        raise


def save_model(model, path="model/xgb_model.pkl"):
    try:
        joblib.dump(model, path)
        logger.info(f"Model saqlandi: {path}")
    except Exception as e:
        logger.error(f"Modelni saqlashda xatolik: {e}")
        raise


if __name__ == "__main__":
    df = load_data("Data/raw/used_car_price_prediction_1M.csv")
    X, y = preprocess(df, target_col="Price")
    model = train_model(X, y)
    save_model(model)
