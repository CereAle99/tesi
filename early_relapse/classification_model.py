import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# load data table
data = pd.read_csv('your_data.csv')
X = data.drop(columns=['EarlyRelapse'])  # Caratteristiche
y = data['EarlyRelapse']  # Etichette

# Divisione dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizzazione delle caratteristiche
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creazione del modello di rete neurale feedforward
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)

# Addestramento del modello
model.fit(X_train, y_train)

# Predizione e valutazione del modello
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
