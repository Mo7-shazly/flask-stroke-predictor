import pandas as pd
import pickle
from flask import Flask, request, jsonify
import os  # 👈 مهم علشان ناخد البورت من متغير البيئة

app = Flask(__name__)

# تحميل النموذج
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
    print("✅ Model loaded successfully!")
    print("أعمدة النموذج الأصلية:", model.feature_names_in_)

# الأعمدة الخام
raw_columns = [
    'id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
    'smoking_status'
]

# الأعمدة المتوقعة (من النموذج)
expected_columns = model.feature_names_in_.tolist()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features', [])

        # 1. التحقق من عدد القيم المدخلة
        if len(features) != len(raw_columns):
            return jsonify({
                "error": f"المتوقع {len(raw_columns)} قيمة، لكن تم إرسال {len(features)}"
            }), 400

        # 2. إنشاء DataFrame بالبيانات المدخلة
        df = pd.DataFrame([features], columns=raw_columns)

        # 3. تطبيق One-Hot Encoding لتحويل القيم النصية إلى رقمية
        df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

        # 4. إضافة الأعمدة المفقودة (إذا كان النموذج يتوقع أعمدة أكثر)
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0

        # 5. ترتيب الأعمدة بنفس ترتيب النموذج وتحويلها إلى أرقام
        df = df[expected_columns].astype(float)

        # 6. طباعة بعض التفاصيل للتأكد
        print("📌 Incoming features:", df.columns.tolist())
        print("📌 Model expects:", expected_columns)
        print("📌 Sample DataFrame:\n", df.head())

        # 7. إجراء التنبؤ
        prediction = model.predict(df)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ تعديل هنا: استخدم البورت من متغير البيئة وشغل على 0.0.0.0
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
