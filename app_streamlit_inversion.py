
import streamlit as st
import pandas as pd
from transformers import pipeline
import yfinance as yf

classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")

st.set_page_config(page_title="IA de Inversión Financiera", layout="wide")

st.title("IA de Inversión basada en Noticias y Riesgo")

# Entrada de usuario
st.sidebar.header("Parámetros de inversión")
capital = st.sidebar.number_input("Capital disponible (€)", min_value=1000, value=10000, step=500)
perfil = st.sidebar.selectbox("Perfil de riesgo", ["conservador", "medio", "agresivo"])
empresas = st.sidebar.multiselect("Empresas a analizar", ["Apple", "Tesla", "Nvidia", "Microsoft", "Google"], default=["Apple", "Tesla"])

st.markdown("---")

# Botón de ejecución
if st.sidebar.button("Analizar y construir cartera"):
    st.subheader("Resultados del Análisis")
    st.write("(Aquí se integrarán los módulos de: análisis de sentimiento, precios, riesgo, correlación y cartera sugerida.)")

    st.markdown("**Cartera sugerida:**")
    cartera_ejemplo = pd.DataFrame({
        "Empresa": ["Apple", "Tesla"],
        "Precio actual": [183.5, 251.2],
        "Acciones": [20, 15],
        "Inversión (€)": [3670, 3768]
    })
    st.dataframe(cartera_ejemplo)

    st.markdown("**Distribución de la inversión:**")
    st.bar_chart(cartera_ejemplo.set_index("Empresa")["Inversión (€)"])

# Backtesting simulado
st.markdown("---")
st.subheader("Backtesting de la estrategia")
if st.button("Ejecutar backtest con noticias simuladas"):
    st.info("Funcionalidad de backtesting real incluida en módulo 'backtesting.py'. Conecta aquí para integrarlo.")

# Carga de noticias personalizadas
st.markdown("---")
st.subheader("Probar IA con tus propias noticias")
archivo = st.file_uploader("Sube un CSV con columnas: fecha, empresa, titulo", type="csv")
if archivo is not None:
    noticias_df = pd.read_csv(archivo)
    st.write("Contenido del archivo cargado:")
    st.dataframe(noticias_df.head())

    resultados = []
    for _, fila in noticias_df.iterrows():
        empresa = fila["empresa"]
        fecha = fila["fecha"]
        texto = fila["titulo"]
        sentimiento = classifier(texto)[0]
        sentimiento_numerico = {"positive": 1, "neutral": 0, "negative": -1}[sentimiento["label"].lower()]

        ticker = yf.Ticker(empresa).info.get("symbol", empresa)
        df = yf.download(ticker, start=fecha, end=pd.to_datetime(fecha) + pd.Timedelta(days=4))
        if df.empty or len(df) < 3:
            continue

        precio_ini = df["Adj Close"].iloc[0]
        precio_fin = df["Adj Close"].iloc[-1]
        retorno = (precio_fin - precio_ini) / precio_ini
        prediccion = "SUBE" if sentimiento_numerico == 1 else "BAJA"
        resultado_real = "SUBE" if retorno > 0 else "BAJA"
        acierto = prediccion == resultado_real

        resultados.append({
            "empresa": empresa,
            "fecha": fecha,
            "noticia": texto,
            "sentimiento": sentimiento["label"],
            "retorno_real": retorno,
            "prediccion": prediccion,
            "resultado": resultado_real,
            "acierto": acierto
        })

    df_resultados = pd.DataFrame(resultados)
    st.markdown("---")
    st.markdown("**Resultados del backtesting personalizado:**")
    st.dataframe(df_resultados)

    if not df_resultados.empty:
        st.metric("Precisión del modelo", f"{df_resultados['acierto'].mean():.2%}")
        st.metric("Rentabilidad media por evento", f"{df_resultados['retorno_real'].mean():.2%}")
