import matplotlib.pyplot as plt
import math
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.stats import skew
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.pipeline import make_pipeline
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score, log_loss, matthews_corrcoef, roc_curve
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
#from skopt import BayesSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline



def uticajni_atributi(podaci_bez_id):
    print(podaci_bez_id.head())

    podaci_bez_id.describe()
    print(podaci_bez_id.dtypes)
    podaci_bez_id.skew()
    numeric_cols = podaci_bez_id.select_dtypes(include=['number']) #diagnosis je objekat a skrew mi radi sa floatom(brojevima) zato sam morao za analizu podataka da odradim bez dijagnoze 
    skew_values = numeric_cols.apply(skew)  #sto je broj veći broj to vise utise na rezultat
    print(skew_values)

def anomalije(podaci_bez_id):
    # Kreiranje modela izolovanog šumskog stabla
    clf = IsolationForest(contamination=0.15)  # contamination predstavlja očekivanu stopu anomaliija, mislim da je 0.2 na ovako mali skup ipak previse
    # Fitovanje modela na podacima
    clf.fit(podaci_bez_id[['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'area_worst', 'perimeter_worst']])

    # Identifikacija anomalija
    outliers = clf.predict(podaci_bez_id[['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'area_worst', 'perimeter_worst']]) == -1

    # Uklanjanje anomalija
    clean_data = podaci_bez_id[~outliers]
    clean_data.to_csv('clean_data.csv')
    
    print("Broj preostalih uzoraka:", clean_data.shape[0])
    
    #print(clean_data.head())
    print("Anomalije izbacene.")
    return clean_data
  
def nezavisne_promenljive(clean_data):
    
    nezavisne_promenljive = ['radius_mean','perimeter_mean', 'area_mean', 'concavity_mean', 
                            'concave points_mean', 'area_worst', 'perimeter_worst']  # da smo zeljeli sve atribute nezavisne_promenljive = podaci.columns.drop('diagnosis')

    plt.figure(figsize=(15, 15))

    for i, atribut in enumerate(nezavisne_promenljive):
        plt.subplot(int(np.ceil(len(nezavisne_promenljive)/5)), 5, i+1) 
        sns.countplot(x=atribut, hue='diagnosis', data=clean_data)
        plt.title(f'{atribut}')
        plt.xlabel(atribut)
        plt.ylabel('Broj pacijenata')
        plt.legend(title='Dijagnoza', loc='upper right')
    plt.tight_layout()
    plt.show()
    
def korelaciona_matrica(clean_data):
    kor_matrica = clean_data.corr()
    #nezavisne_promenljive(clean_data)
    plt.figure(figsize=(12, 10))
    sns.heatmap(kor_matrica, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Matrica korelacije atributa')
    plt.show()

def  korelaciona_matrica_Znacajni(clean_data):
    
    kor_matrica = clean_data.corr()
    odabrani_atributi = clean_data[['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'area_worst', 'perimeter_worst']]
    # Izračunavanje korelacione matrice samo za odabrane atribute
    korelaciona_matrica_odabranih_atributa = odabrani_atributi.corr()
    
    print(korelaciona_matrica_odabranih_atributa)
    plt.figure(figsize=(10, 8))
    sns.heatmap(korelaciona_matrica_odabranih_atributa, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelaciona matrica znacajnih atributa')
    plt.show()
    
def balansiranje_undersampling(clean_data):
    indeksi_malignih = clean_data[clean_data['diagnosis'] == 1].index
    indeksi_benignih = clean_data[clean_data['diagnosis'] == 0].index

    # Odabir nasumičnih indeksa benignih uzoraka kako bi se izjednačila sa brojem malignih uzoraka
    broj_malignih = len(indeksi_malignih)
    indeksi_benignih_undersampled = np.random.choice(indeksi_benignih, size=broj_malignih, replace=False)

    # Spajanje indeksa uzoraka oba razreda
    balansirani_indeksi = np.concatenate([indeksi_malignih, indeksi_benignih_undersampled])

    # Izdvajanje balansiranih podataka
    balansirani_podaci = clean_data.loc[balansirani_indeksi]

    #print(balansirani_podaci.head())
    print("podaci su balansirani")
    return balansirani_podaci

def balansiranje_oversampling(clean_data):
    # Izdvajanje indeksa uzoraka za svaku klasu
    indeksi_malignih = clean_data[clean_data['diagnosis'] == 1].index
    indeksi_benignih = clean_data[clean_data['diagnosis'] == 0].index

    # Odabir nasumičnih indeksa malignih uzoraka kako bi se izjednačila sa brojem benignih uzoraka
    broj_benignih = len(indeksi_benignih)
    indeksi_malignih_oversampled = np.random.choice(indeksi_malignih, size=broj_benignih, replace=True)

    # Spajanje indeksa uzoraka oba razreda
    balansirani_indeksi = np.concatenate([indeksi_benignih, indeksi_malignih_oversampled])

    # Izdvajanje balansiranih podataka
    balansirani_podaci = clean_data.loc[balansirani_indeksi]
    print("podaci su balansirani")
    return balansirani_podaci
    # Izdvojeni podaci su sada balansirani

def KNN_metoda_lakta(X_train_scaled,X_test_scaled,y_train,y_test):
    # Metoda lakta za određivanje optimalnog broja suseda
    error_rates = []
    k_range = range(1, 11)
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_predict = knn.predict(X_test_scaled)
        error = np.mean(y_predict != y_test)
        error_rates.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, error_rates, marker='o', linestyle='--')
    plt.xlabel('Broj suseda (k)')
    plt.ylabel('Stopa greške')
    plt.title('Metoda lakta za optimalan broj suseda')
    plt.show()
    
    # Biranje najboljeg k na osnovu metode lakta
    optimal_k = k_range[np.argmin(error_rates)]
    print(f"Optimalan broj suseda je: {optimal_k}")
    
    # Evaluacija modela sa optimalnim k
    knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_optimal.fit(X_train_scaled, y_train)
    y_predict = knn_optimal.predict(X_test_scaled)
    
    print(f"Rezultati za KNN sa k={optimal_k}:")
    print("Tačnost je: ", accuracy_score(y_test, y_predict))
    print("Preciznost je: ", precision_score(y_test, y_predict))
    print("Osjetljivost: ", recall_score(y_test, y_predict))
    print("F1: ", f1_score(y_test, y_predict))
    print("Matrica konfuzije:")
    print(confusion_matrix(y_test, y_predict))
    print("--------------------------------------")    
     
def tipovi_grafova(clean_data):
    # Linijski grafik
    sns.lineplot(x='area_mean', y='perimeter_mean', hue='diagnosis', data=clean_data)

    # Postavljanje naslova i oznaka osa
    plt.title('Linijski grafik - area_mean vs. perimeter_mean')
    plt.xlabel('area_mean')
    plt.ylabel('perimeter_mean')

    # Poboljšavanje rasporeda elemenata
    plt.tight_layout()
    plt.show()

    # Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='radius_mean', y='area_mean', hue='diagnosis', data=clean_data)
    plt.title('Scatter plot - radius_mean vs. area_mean')
    plt.xlabel('radius_mean')
    plt.ylabel('area_mean')
    plt.legend(title='Dijagnoza', loc='upper right')
    plt.show()

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='diagnosis', y='perimeter_mean', data=clean_data)
    plt.title('Boxplot - perimeter_mean')
    plt.xlabel('Dijagnoza')
    plt.ylabel('perimeter_mean')
    plt.show()

    # Bar plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis', data=clean_data)
    plt.title('Bar plot - dijagnoza')
    plt.xlabel('Dijagnoza')
    plt.ylabel('Broj pacijenata')
    plt.show()

    # Line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='area_mean', y='concavity_mean', hue='diagnosis', data=clean_data)
    plt.title('Line plot - area_mean vs. concavity_mean')
    plt.xlabel('area_mean')
    plt.ylabel('concavity_mean')
    plt.legend(title='Dijagnoza', loc='upper right')
    plt.show()

    # Histogram
    plt.subplot(1, 3, 1)
    sns.histplot(clean_data['radius_mean'], bins=20, kde=True, color='skyblue')
    plt.title('Histogram - radius_mean')
    plt.xlabel('radius_mean')
    plt.ylabel('Broj pacijenata')

    # Pie grafik
    plt.subplot(1, 3, 2)
    clean_data['diagnosis'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('Pita grafik - dijagnoza')
    plt.ylabel('')

    # Violin plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='diagnosis', y='perimeter_worst', data=clean_data)
    plt.title('Violin plot - perimeter_worst')
    plt.xlabel('Dijagnoza')
    plt.ylabel('perimeter_worst')
    plt.show()
 
def roc_kriva(fpr, tpr, auc, model_name):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def stacking(X_train_scaled,X_test_scaled,y_train,y_test):
        # Bazni model - Random Forest Classifier
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train_scaled, y_train)

    # Izračunavanje predikcija baznog modela na test skupu
    base_predictions = base_model.predict(X_test_scaled)

    # Metamodel - Logistic Regression
    meta_model = LogisticRegression()
    meta_model.fit(base_predictions.reshape(-1, 1), y_test)  # Koristimo predikcije baznog modela kao ulazne značajke

    # Izračunavanje predikcija metamodela
    stacked_predictions = meta_model.predict(base_predictions.reshape(-1, 1))

    # Evaluacija performansi stacked modela
    accuracy = accuracy_score(y_test, stacked_predictions)
    return print("Accuracy of stacked model:", accuracy)
    # U ovom primjeru, koristimo Random Forest kao bazni model za generiranje predikcija na test skupu. Zatim, koristimo te predikcije kao ulazne značajke za logističku regresiju, koja je metamodel. Na kraju, 
    # koristimo metamodel za predikciju.
    # Ova tehnika može se dalje proširiti dodavanjem više različitih baznih modela i/ili podešavanjem parametara metamodela kako bi se postigli bolji rezultati.

def beging(X_train_scaled,X_test_scaled,y_train,y_test):
    # Definiranje baznog modela - stablo odlučivanja
    base_model = DecisionTreeClassifier() 

    # Inicijalizacija i treniranje bagging klasifikatora
    bagging_model = BaggingClassifier(base_model, n_estimators=10, random_state=42)
    bagging_model.fit(X_train_scaled, y_train)

    # Predikcija na testnom skupu
    y_predict = bagging_model.predict(X_test_scaled)

    # Evaluacija performansi bagging modela
    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy of bagging model:", accuracy)

def boosting(X_train_scaled,X_test_scaled,y_train,y_test):
    # Inicijalizacija i treniranje AdaBoost klasifikatora
    boosting_model = AdaBoostClassifier(n_estimators=50, random_state=42)
    boosting_model.fit(X_train_scaled, y_train)

    # Predikcija na testnom skupu
    y_predict = boosting_model.predict(X_test_scaled)

    # Evaluacija performansi boosting modela
    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy of boosting model:", accuracy)

def podasavanje_hiperparametara_Grid_Search_Cross_Validation(X_train_scaled,X_test_scaled,y_train,y_test):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.5, 1.0],
    }

    # Inicijalizacija AdaBoost modela
    adaboost_model = AdaBoostClassifier(algorithm='SAMME', random_state=42)
    
    # Inicijalizacija Grid Search Cross-Validation objekta
    grid_search = GridSearchCV(estimator=adaboost_model, param_grid=param_grid, cv=5)
    
    # Pokretanje pretrage
    grid_search.fit(X_train_scaled, y_train)
    
    # Ispis najboljih hiperparametara
    print("Najbolji hiperparametri:", grid_search.best_params_)
    
    # Dobivanje najboljeg modela
    best_model = grid_search.best_estimator_
    
    # Evaluacija performansi najboljeg modela
    y_predict = best_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_predict)
    print("Accuracy of best model:", accuracy)

def podesavanje_hiperparametara_Randomized_Search_Cross_Validation(X_train_scaled,X_test_scaled,y_train,y_test):
    # Definiranje hiperparametara koje želimo testirati za KNeighborsClassifier
    param_grid = {
        'n_neighbors': [3, 5, 7],   #radio sam i sa randint(1, 10),
        'weights': ['uniform', 'distance']
    }
    # Inicijalizacija KNeighborsClassifier modela
    knn_model = KNeighborsClassifier()

    # Inicijalizacija Grid Search Cross-Validation objekta
    grid_search_knn = GridSearchCV(estimator=knn_model, param_grid=param_grid, cv=5)

    # Pokretanje pretrage
    grid_search_knn.fit(X_train_scaled, y_train)

    # Ispis najboljih hiperparametara
    print("Najbolji hiperparametri za KNeighborsClassifier:", grid_search_knn.best_params_)

    # Dobivanje najboljeg modela za KNeighborsClassifier
    best_knn_model = grid_search_knn.best_estimator_

    # Evaluacija performansi najboljeg modela za KNeighborsClassifier
    y_pred_knn = best_knn_model.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    print("Accuracy of best KNeighborsClassifier model:", accuracy_knn)


def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = []
    for model in models:
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        result = {
            'Model': model.__class__.__name__,
            'Accuracy': accuracy_score(y_test, y_predict),
            'Precision': precision_score(y_test, y_predict),
            'Recall': recall_score(y_test, y_predict),
            'F1 Score': f1_score(y_test, y_predict)
        }

        if y_prob is not None:
            result['ROC-AUC'] = roc_auc_score(y_test, y_prob)
            result['Log Loss'] = log_loss(y_test, y_prob)
            #pr, tpr, _ = roc_curve(y_test, y_prob)  #ove dve linije za plotovanje roc_krive
            #roc_kriva(pr, tpr, roc_auc_score(y_test, y_prob), model.__class__.__name__)

        results.append(result)

    return pd.DataFrame(results)

          


def main():
        
    podaci = pd.read_csv('Cancer_Data.csv')
    podaci['diagnosis'] = podaci['diagnosis'].map({'M': 1, 'B': 0})
    podaci.isnull().any()
    podaci.diagnosis.unique()
    #print("Duplicirani podaci: ", podaci.duplicated().sum())  #duplikati? nemam dupliciranih podatak
    podaci_bez_id = podaci.drop('id', axis=1)
    podaci_bez_id = podaci_bez_id.drop(columns=['Unnamed: 32']) #brisem unnamed kolonu em ne znam kako se pojavila em sam pola zivota izgubio sa njom i Nan vrednostima koje se magicno pojavljuju zivota mi, ovo je svojevrsni rage quit da se zna!!!!!
    
    #podaci_bez_id.info()

    #diag_gr = podaci_bez_id.groupby('diagnosis')
    #print("Broj malignih uzoraka:", (podaci_bez_id['diagnosis'] == 1).sum())
    #print("Broj beningnih uzoraka:", (podaci_bez_id['diagnosis'] == 0).sum())

    #uticajni_atributi(podaci_bez_id)
    clean_data = anomalije(podaci_bez_id)
    #korelaciona_matrica(clean_data)
    #korelaciona_matrica_Znacajni(clean_data)
    
 

    
    #X = clean_data[['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 'area_worst', 'perimeter_worst']]
    #y=clean_data['diagnosis']
    X = clean_data.drop(columns=['diagnosis'],axis=1)
    y = clean_data['diagnosis']
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y,random_state=42)   #random_state=42 osiguravamo da ćemo svaki put dobiti iste rezultate




    #----------------------------------------------Balansiranje-------------------------------------
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    #clean_data=balansiranje_undersampling(clean_data) s obzirom na broj podataka mislim da nije pametno dodatno smanjivati broj uzoraka
    #clean_data=balansiranje_oversampling(clean_data)
    print(f'Klasa pre balansiranja: {Counter(y_train)}')
    print(f'Klase nakon balansiranja: {Counter(y_train_balanced)}')
    #-----------------------------------------------------------------------------------------------
    
    
    #---------------------------------------------Skaliranje-----------------------------------------
    #Skaliranje podataka (opcionalno, ali često korisno za algoritme poput SVM-a)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    #------------------------------------------------------------------------------------------------
    
    
    #--------------------------------Selektovanje najbitnijih atributa-------------------------------    
    # Model-Based Feature Selection sa Random Forestom (za poslednju tacku)
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
    selector.fit(X_train_scaled, y_train_balanced)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    #-------------------------------------------------------------------------------------------------
    
    
    
    #------------------------------------------Modeli-------------------------------------------------
    models = [KNeighborsClassifier(), LogisticRegression(max_iter=10000), SVC(probability=True), RandomForestClassifier(), 
                GradientBoostingClassifier(), AdaBoostClassifier(), GaussianNB(), DecisionTreeClassifier(), 
                MLPClassifier(max_iter=10000)]
   
    print("----------------------------") #za ljepsi ispis
    #--------------------------------------------------------------------------------------------------
    
    KNN_metoda_lakta(X_train_scaled,X_test_scaled,y_train_balanced,y_test)
    podesavanje_hiperparametara_Randomized_Search_Cross_Validation(X_train_scaled,X_test_scaled,y_train_balanced,y_test)
    podasavanje_hiperparametara_Grid_Search_Cross_Validation(X_train_scaled,X_test_scaled,y_train_balanced,y_test)
    
    
    
    
    #--------------------------------------REZULTATI------------------------------------------------

    # Evaluacija modela sa svim atributima
    print("Rezultati sa svim atributima:")
    svi_atributi=evaluate_models(models,X_train_scaled, X_test_scaled, y_train_balanced, y_test)
    print(svi_atributi) 
    
    # Evaluacija modela sa odabranim atributima
    print("\nRezultati sa odabranim atributima:")
    odabrani_atributi=evaluate_models(models,X_train_selected, X_test_selected, y_train_balanced, y_test)      
    print(odabrani_atributi) 
    
    # Kreiranje DataFrame za rezultate
    df_all_features = pd.DataFrame(svi_atributi)
    df_selected_features = pd.DataFrame(odabrani_atributi)

    # Spajanje rezultata u jednu tabelu za lakše poređenje
    df_results = pd.merge(df_all_features, df_selected_features, on='Model', suffixes=('_all_features', '_selected_features'))

    # Prikazivanje rezultata u tabelarnoj formi
    print("\nUporedni rezultati modela:")
    print(df_results)
    
    #----------------------------------------------------------------------------------------------------
    
    stacking(X_train_scaled,X_test_scaled,y_train_balanced,y_test)   
    beging(X_train_scaled,X_test_scaled,y_train_balanced,y_test)
    boosting(X_train_scaled,X_test_scaled,y_train_balanced,y_test)
    
    

    for model in models:
        pipeline = make_pipeline(StandardScaler(), model)
        # Unakrsna validacija sa 5 preklopa
        scores = cross_val_score(pipeline, X_train_scaled, y_train_balanced, cv=5, scoring='accuracy')
        
        # Ispis rezultata unakrsne validacije
        print("Model:", model.__class__.__name__)
        print("Srednja tačnost:", scores.mean())
        print("--------------------------------------")
    #tipovi_grafova(clean_data)

 



        

if __name__ == "__main__":
    main()