import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import os
import matplotlib.pyplot as plt

# Aseta sivun otsikko
st.set_page_config(page_title="Koripallo XGBoost Analysaattori", layout="wide")
st.title("Koripallo XGBoost Analysaattori")

# Alusta istuntomuuttujat, jos niitä ei ole vielä
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.feature_names = None
    st.session_state.original_data = None
    st.session_state.training_columns = None
    st.session_state.unique_values = {}
    st.session_state.columns = []

# Lataa malli ja data, jos niitä ei ole vielä ladattu
@st.cache_resource
def load_model_and_data():
    model = None
    feature_names = None
    original_data = None
    training_columns = None
    unique_values = {}
    columns = []
    
    # Yritä ladata malli
    try:
        model = pickle.load(open("koripallo_xgboost_malli.pkl", "rb"))
        st.success("Malli ladattu onnistuneesti!")
        
        # Yritä ladata sarakkeiden nimet
        try:
            feature_names = pickle.load(open("feature_names.pkl", "rb"))
            st.success("Sarakkeiden nimet ladattu onnistuneesti!")
        except:
            st.warning("Sarakkeiden nimiä ei löytynyt. Luodaan ne alkuperäisestä datasta.")
    except:
        st.error("Mallia ei löytynyt. Lataa malli ennen ennusteiden tekemistä.")
    
    # Lataa alkuperäinen data
    try:
        original_data = pd.read_csv('siivottuja_4.csv', sep=';', decimal=',')
        training_columns = list(original_data.columns)
        training_columns.remove('XP')  # Poista XP
        
        # Jos ei löydy sarakkeiden nimiä, luodaan ne
        if feature_names is None and original_data is not None:
            # Muunnetaan data kategorisiksi ja tehdään one-hot koodaus
            X_categorical = original_data.drop('XP', axis=1).astype('category')
            X_dummies = pd.get_dummies(X_categorical)
            feature_names = list(X_dummies.columns)
            
            # Tallennetaan sarakkeiden nimet tulevaa käyttöä varten
            pickle.dump(feature_names, open("feature_names.pkl", "wb"))
            st.success("Sarakkeiden nimet luotu ja tallennettu.")
        
        # Kerää jokaisen sarakkeen uniikit arvot
        columns = list(original_data.columns)
        columns.remove('XP')  # Poista XP
        for col in columns:
            unique_values[col] = sorted(original_data[col].unique().astype(str))
        
    except Exception as e:
        st.error(f"Virhe datan latauksessa: {str(e)}")
        # Määritellään esimerkkisarakkeet, jos dataa ei ole saatavilla
        columns = ["Team", "Set", "Opp.Defence", "Ballscreen", "Ballscreen defense", 
                  "Line up", "Team fouls", "Point difference", "Shot clock (half court)", 
                  "Time", "Offence start", "Shot", "Shot clock", "Player"]
        unique_values = {col: ["1", "2", "3"] for col in columns}
        st.warning("Alkuperäistä dataa ei löytynyt. Käytetään esimerkkiarvoja.")
    
    return model, feature_names, original_data, training_columns, unique_values, columns

# Lataa malli ja data
model, feature_names, original_data, training_columns, unique_values, columns = load_model_and_data()

# Päivitä istuntomuuttujat
st.session_state.model = model
st.session_state.feature_names = feature_names
st.session_state.original_data = original_data
st.session_state.training_columns = training_columns
st.session_state.unique_values = unique_values
st.session_state.columns = columns

# Funktio syötteen valmisteluun ennustusta varten
def prepare_input_for_prediction(selected_data, feature_names):
    # Luo DataFrame valituista tiedoista
    input_df = pd.DataFrame([selected_data])
    
    # Muunna kategorisiksi
    input_categorical = input_df.astype('category')
    
    # Tee one-hot koodaus
    input_dummies = pd.get_dummies(input_categorical)
    
    # Tarkista, että kaikki koulutusdatan sarakkeet ovat käytössä
    if feature_names is not None:
        # Luo uusi DataFrame, joka sisältää kaikki koulutusdatan sarakkeet
        final_df = pd.DataFrame(0, index=input_dummies.index, columns=feature_names)
        
        # Kopioi olemassa olevat sarakkeet
        for col in input_dummies.columns:
            if col in feature_names:
                final_df[col] = input_dummies[col]
            else:
                st.warning(f"Varoitus: Saraketta {col} ei löydy koulutusdatasta.")
        
        return final_df
    else:
        # Jos ei ole tietoa koulutusdatan sarakkeista, käytä suoraan syötettä
        return input_dummies

# Välilehdet
tab1, tab2 = st.tabs(["XP Ennuste", "Paras Set"])

# Välilehti 1: XP Ennuste
with tab1:
    st.header("Valitse muuttujat ja niiden arvot XP-ennustetta varten")
    
    # Käytä kolumneja asetteluun
    col1, col2 = st.columns(2)
    
    # Luo valintalaatikot ja pudotusvalikot
    selected_data = {}
    
    # Jaa sarakkeet kahdelle puolelle
    mid_point = len(st.session_state.columns) // 2
    
    with col1:
        for i, col in enumerate(st.session_state.columns[:mid_point]):
            st.subheader(col)
            use_feature = st.checkbox(f"Käytä {col}", value=True, key=f"use_{col}")
            if use_feature:
                if st.session_state.unique_values.get(col):
                    value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_{col}")
                    selected_data[col] = int(value) if value.isdigit() else value
    
    with col2:
        for i, col in enumerate(st.session_state.columns[mid_point:]):
            st.subheader(col)
            use_feature = st.checkbox(f"Käytä {col}", value=True, key=f"use_{col}")
            if use_feature:
                if st.session_state.unique_values.get(col):
                    value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_{col}")
                    selected_data[col] = int(value) if value.isdigit() else value
    
    # Ennustuspainike
    if st.button("Ennusta XP", key="predict_button"):
        if st.session_state.model is None:
            st.error("Mallia ei ole ladattu. Lataa malli ensin.")
        else:
            # Valmistele syöte ennustusta varten
            input_dummies = prepare_input_for_prediction(selected_data, st.session_state.feature_names)
            
            # Tee ennuste
            try:
                dmatrix = xgb.DMatrix(input_dummies)
                prediction = st.session_state.model.predict(dmatrix)[0]
                
                # Näytä tulos
                st.success(f"Ennustettu XP: {prediction:.3f}")
            except Exception as e:
                st.error(f"Virhe ennustetta tehdessä: {str(e)}")
                st.error(f"Tarkempi virhetieto: {str(e)}")

# Välilehti 2: Paras Set
with tab2:
    st.header("Etsi paras Set muiden muuttujien perusteella")
    
    # Käytä kolumneja asetteluun
    col1, col2 = st.columns(2)
    
    # Luo valintalaatikot ja pudotusvalikot, paitsi Set
    selected_data_set = {}
    
    # Jaa sarakkeet kahdelle puolelle
    non_set_columns = [col for col in st.session_state.columns if col != "Set"]
    mid_point = len(non_set_columns) // 2
    
    with col1:
        for i, col in enumerate(non_set_columns[:mid_point]):
            st.subheader(col)
            use_feature = st.checkbox(f"Käytä {col}", value=True, key=f"use_set_{col}")
            if use_feature:
                if st.session_state.unique_values.get(col):
                    value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_set_{col}")
                    selected_data_set[col] = int(value) if value.isdigit() else value
    
    with col2:
        for i, col in enumerate(non_set_columns[mid_point:]):
            st.subheader(col)
            use_feature = st.checkbox(f"Käytä {col}", value=True, key=f"use_set_{col}")
            if use_feature:
                if st.session_state.unique_values.get(col):
                    value = st.selectbox(f"Valitse {col}", st.session_state.unique_values[col], key=f"select_set_{col}")
                    selected_data_set[col] = int(value) if value.isdigit() else value
    
    # Etsintäpainike
    if st.button("Etsi paras Set", key="find_best_set"):
        if st.session_state.model is None:
            st.error("Mallia ei ole ladattu. Lataa malli ensin.")
        else:
            # Löydä kaikki mahdolliset Set-arvot
            set_values = st.session_state.unique_values.get("Set", [])
            
            if not set_values:
                st.error("Set-arvoja ei löytynyt.")
            else:
                # Tee ennusteet jokaiselle Set-arvolle
                results = []
                
                with st.spinner("Lasketaan parasta Settiä..."):
                    for set_val in set_values:
                        # Luo kopio valituista tiedoista ja lisää nykyinen Set-arvo
                        current_data = selected_data_set.copy()
                        current_data["Set"] = int(set_val) if set_val.isdigit() else set_val
                        
                        # Valmistele syöte ennustusta varten
                        input_dummies = prepare_input_for_prediction(current_data, st.session_state.feature_names)
                        
                        # Tee ennuste
                        try:
                            dmatrix = xgb.DMatrix(input_dummies)
                            prediction = st.session_state.model.predict(dmatrix)[0]
                            results.append((set_val, prediction))
                        except Exception as e:
                            st.error(f"Virhe ennustetta tehdessä Set-arvolle {set_val}: {str(e)}")
                            st.error(f"Tarkempi virhetieto: {str(e)}")
                            break
                
                # Jos tuloksia löytyi, visualisoi ne
                if results:
                    # Järjestä tulokset XP-ennusteen mukaan
                    results.sort(key=lambda x: x[1], reverse=True)
                    
                    # Erottele data
                    set_values = [str(r[0]) for r in results]
                    xp_values = [r[1] for r in results]
                    
                    # Näytä paras tulos
                    best_set, best_xp = results[0]
                    st.success(f"Paras Set: {best_set} (XP: {best_xp:.3f})")
                    
                    # Luo pylväsdiagrammi
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(set_values, xp_values, color='skyblue')
                    
                    # Korosta paras tulos
                    best_idx = xp_values.index(max(xp_values))
                    bars[best_idx].set_color('green')
                    
                    # Lisää arvot palkkien päälle
                    for bar, xp in zip(bars, xp_values):
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                               f"{xp:.3f}", ha='center', va='bottom')
                    
                    # Muotoile kuvaaja
                    ax.set_xlabel('Set')
                    ax.set_ylabel('Ennustettu XP')
                    ax.set_title('Eri Set-arvojen ennustetut XP-arvot')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Näytä kuvaaja
                    st.pyplot(fig)

# Tietoja sovelluksesta
with st.expander("Tietoja sovelluksesta"):
    st.write("""
    ## Koripallo XGBoost Analysaattori
    
    Tämä sovellus käyttää XGBoost-koneoppimismallia koripallon pelianalyysiin.
    
    Sovellus mahdollistaa:
    - XP-arvojen ennustamisen eri pelitilanteissa
    - Parhaan Set-vaihtoehdon löytämisen valituilla muuttujilla
    
    Alkuperäinen sovellus on muunnettu Streamlitiin, jotta sitä voidaan käyttää verkkoselaimessa.
    """)
