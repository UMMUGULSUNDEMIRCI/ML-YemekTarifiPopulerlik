# Gerekli kütüphaneleri içe aktarıyoruz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CSV dosyasını okuyoruz
df = pd.read_csv("C:\\Users\\demir\\Desktop\\yemek_tarif.csv")

df.info()
df.head()

#Ilk sutun kategorik verilerin ele alınması
# One-Hot Encoding
df = pd.get_dummies(df, columns=['porsiyon'])
df.head()

# İkinci sütunu düzenle
def hsure_duzelt(time_str):
    # Saat ve dakika birimlerini sırasıyla işleyelim
    hours = 0
    minutes = 0
    if 'saat' in time_str:
        hours = int(time_str.split(' saat')[0])  # Saat değerini al
    if 'dk' in time_str:
        minutes = int(time_str.split('dk')[0].split()[-1])  # Dakika değerini al
    total_minutes = hours * 60 + minutes  # Toplam dakika
    return total_minutes
    
# Fonksiyonu ikinci sütuna uygula
df["hazirlik_suresi"] = df["hazirlik_suresi"].apply(hsure_duzelt)

# Sonuçları yazdıralım
print(df)

# ucuncu sütunu düzenle
def psure_duzelt(time_str):
    # Saat ve dakika birimlerini sırasıyla işleyelim
    hours = 0
    minutes = 0
    if 'saat' in time_str:
        hours = int(time_str.split(' saat')[0])  # Saat değerini al
    if 'dk' in time_str:
        minutes = int(time_str.split('dk')[0].split()[-1])  # Dakika değerini al
    total_minutes = hours * 60 + minutes  # Toplam dakika
    return total_minutes
    
df["pisirme_suresi"] = df["pisirme_suresi"].apply(psure_duzelt)

# Düzenlenmiş DataFrame'i yazdır
print(df)


#dorduncu sutunu duzelt
def defter_duzelt(deger):
    try:
        # Metni küçük harfe çevir ve gereksiz kelimeleri kaldır
        deger = deger.lower().strip()
        if "bin" in deger:
            # bin  olanları 000 e cevir
            binli = int(deger.replace("bin", "").strip())
            return binli * 1000
        elif deger.isdigit():
            # Eğer sadece sayi varsa
            return int(deger)
        else:
            return None
    except Exception as e:
        print(f"Hata oluştu: {e}")  # Hata mesajını yazdır
        return None
df["deftere_ekleme"] = df["deftere_ekleme"].apply(defter_duzelt)

# Düzenlenmiş DataFrame'i yazdır
print(df)

df.head()
df.info()

# Veri setinizdeki boş hücre sayısını kontrol etme
empty_columns = df.isna().sum()

# Boş sütunları ve her bir sütundaki eksik değerlerin sayısını yazdırma
print(empty_columns)

# Sayısal sütunları seçelim
numeric_columns = df.select_dtypes(include='number').columns

# Grafik boyutunu ayarlayalım
plt.figure(figsize=(15, 12))

# Sayısal sütunlar için histogramları çizelim
for i, col in enumerate(numeric_columns):
    plt.subplot(3, 4, i+1)  # 3 satır 4 sütun şeklinde düzenle
    sns.histplot(df[col], kde=True, bins=20, color='purple')  # Histogram ve Kernel Density Estimation (KDE) çizecek
    plt.title(f'Histogram of {col}')
    plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt

# Boolean türündeki sütunları seçelim
boolean_columns = df.select_dtypes(include='bool').columns

# Her bir Boolean sütununda True yüzdesini hesaplayalım
true_percentages = [df[col].mean() * 100 for col in boolean_columns]  # True değerlerinin yüzdesi

# Diğer kategori (False) yüzdesini hesaplayalım
false_percentages = [100 - p for p in true_percentages]  # False yüzdesi, 100 - True yüzdesi

# Kullanacağımız renk paletini belirleyelim (8 farklı renk)
colors = ['#FF6347', '#FF4500', '#FFD700', '#00BFFF', 'darkblue', '#ADFF2F', '#8A2BE2', '#FF1493']

# Grafik boyutunu ayarlayalım
plt.figure(figsize=(12, 8))  # Grafik boyutunu artırdık

# True yüzdeleri için pasta grafiği
plt.pie(true_percentages, autopct='%1.1f%%', colors=colors, startangle=90)

# Başlık
plt.title('True Value Percentages Across All Boolean Columns')

# Legend ekleyelim
plt.legend(labels=boolean_columns, title='Kategorik Sütunlar', loc='upper left', fontsize=10, bbox_to_anchor=(1, 1))

# Grafiği eşit oranda hizalayalım
plt.axis('equal')  

# Layoutu sıkılaştırarak görüntüyü iyileştirelim
plt.tight_layout()
plt.show()

# Sayısal sütunları seçelim
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Korelasyon matrisini hesaplayalım
correlation_matrix = df[numeric_columns].corr()

# Korelasyon matrisini görselleştirelim
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Korelasyon Matrisi')
plt.show()

import numpy as np
# Aykırı değerleri tespit etmek için IQR hesaplayalım
def detect_outliers_iqr(df):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)  # 1. çeyrek
        Q3 = df[col].quantile(0.75)  # 3. çeyrek
        IQR = Q3 - Q1  # IQR (Interquartile Range)

        lower_bound = Q1 - 1.5 * IQR  # Alt sınır
        upper_bound = Q3 + 1.5 * IQR  # Üst sınır

        # Aykırı değerlerin bulunduğu satırları tespit et
        outliers_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outliers[col] = outliers_col

    return outliers  # Aykırı değerlerin bulunduğu satırları sütun bazında döndürelim

# Aykırı değerlerin bulunduğu satırları tespit edelim
outliers = detect_outliers_iqr(df)
print(f"Aykırı değerlerin bulunduğu satır sayısı: {sum(len(v) for v in outliers.values())}")

# Aykırı değerleri ortalama ile doldururken veri tiplerini uyumlulaştırma
for column in df.select_dtypes(include=['float64', 'int64']).columns:
    outlier_rows = outliers.get(column, [])
    if len(outlier_rows) > 0:  # Aykırı değer varsa
        mean_value = df[column].mean()

        # Hedef sütunun veri tipine uygun şekilde dönüştürme
        if pd.isna(mean_value):  # Eğer mean_value NaN ise
            continue  # NaN ise atla
        if df[column].dtype == 'int64':
            mean_value = int(mean_value)  # float değeri int'ye dönüştür
        df.loc[outlier_rows, column] = mean_value

print("Güncellenmiş DataFrame:")
print(df)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Korelasyon eşik değeri
correlation_threshold = 0.05

# Hedef sütun ile korelasyon hesaplama
correlations = df.corr(method='spearman')['deftere_ekleme'].drop('deftere_ekleme')  # Hedef sütun hariç
low_corr_columns = correlations[correlations.abs() < correlation_threshold].index.tolist()

# Çıkarılan sütunları yazdırma
print(f"Çıkarılan sütunlar: {low_corr_columns}")

# Düşük korelasyonlu sütunları çıkarma
df_cleaned = df.drop(columns=low_corr_columns)
df = df_cleaned.copy()

# Özellik ve hedef sütunları ayırın
X = df.drop(['deftere_ekleme'], axis=1)  # Özellikler
y = df['deftere_ekleme']  # Hedef

# Hedef değişken (y) üzerindeki eksik değerleri kontrol et
print(y.isna().sum())  # Eksik (NaN) değerlerin sayısını yazdır
# Hedef değişkendeki eksik değerleri ortalama ile doldur
y = y.fillna(y.mean())

y = np.log1p(y)  # Hedef değişkene log dönüşümü

# Eğitim ve test setine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Özellik ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Sadece eğitim setini fit edin
X_test_scaled = scaler.transform(X_test)       # Test setini sadece transform edin

# Model değerlendirme fonksiyonu
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

   
    # Gerçek değerler (test seti) ile tahmin edilen değerleri çizelim
    plt.figure(figsize=(5, 3))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Tahminler')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Doğru Değerler')
    plt.title(f'{model_name} - Gerçek ve Tahmin Edilen Değerler')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.legend()
    plt.show()

    print(f"{model_name} - Mean Squared Error: {mse:.4f}")
    print(f"{model_name} - R^2 Score: {r2:.4f}")


# Random Forest Modeli
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train, y_train, X_test, y_test, "Random Forest")

# Gradient Boosting Modeli
gb_model = GradientBoostingRegressor(random_state=42)
evaluate_model(gb_model, X_train, y_train, X_test, y_test, "Gradient Boosting")

# XGBoost Modeli
xgb_model = xgb.XGBRegressor(random_state=42)
evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost")


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split



# Bins düzenlemesi
bins = [0, 2000, 10000, 21000, 60000]
labels = ['düşük', 'orta', 'yüksek', 'çok yüksek']

# Yeni kategori sütununu oluşturma
df['deftere_ekleme_kategori'] = pd.cut(df['deftere_ekleme'], bins=bins, labels=labels, right=True)

# Kategorilerin sayısını görmek için
print(df['deftere_ekleme_kategori'].value_counts())

# Özellikler (X) ve hedef değişken (y)
X = df.drop('deftere_ekleme_kategori', axis=1)  # Bağımsız değişkenler
y = df['deftere_ekleme_kategori']  # Bağımlı değişken

# LabelEncoder ile sınıf etiketlerini sayısal değerlere dönüştürme
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Veriyi eğitim ve test setlerine bölme (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Veriyi normalize etme (SVM için önemli)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# SVM modeli
svm_model = SVC()

# SVM parametreleri için GridSearchCV kullanarak en iyi parametreyi bulma
svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
svm_grid_search = GridSearchCV(svm_model, svm_params, cv=5, scoring='accuracy')
svm_grid_search.fit(X_train_scaled, y_train)

# En iyi parametreleri yazdırma
print("SVM En İyi Parametreler:", svm_grid_search.best_params_)

# KNN modeli
knn_model = KNeighborsClassifier()

# KNN parametreleri için GridSearchCV kullanarak en iyi parametreyi bulma
knn_params = {'n_neighbors': [3, 5, 7, 9], 'metric': ['euclidean', 'manhattan']}
knn_grid_search = GridSearchCV(knn_model, knn_params, cv=5, scoring='accuracy')
knn_grid_search.fit(X_train_scaled, y_train)

# En iyi parametreleri yazdırma
print("KNN En İyi Parametreler:", knn_grid_search.best_params_)

# SVM modelini oluşturma ve eğitme (C=10, gamma='scale')
svm_model = SVC(C=10, gamma='scale', kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# SVM ile tahmin yapma
y_pred_svm = svm_model.predict(X_test_scaled)

# SVM sonuçlarını yazdırma
print("SVM Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_svm, target_names=label_encoder.classes_))
# Confusion matrisini oluşturma ve görselleştirme
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('SVM Confusion Matris')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

# KNN modelini oluşturma (metric='manhattan', n_neighbors=9)
knn_model = KNeighborsClassifier(n_neighbors=9, metric='manhattan')
knn_model.fit(X_train_scaled, y_train)

# KNN ile tahmin yapma
y_pred_knn = knn_model.predict(X_test_scaled)

# KNN sonuçlarını yazdırma
print("KNN Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_knn, target_names=label_encoder.classes_))

# Confusion matrisini oluşturma ve görselleştirme
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('KNN Confusion Matris')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
