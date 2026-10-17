import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Configuração da página do Streamlit
st.set_page_config(
    page_title="Previsão de Consumo de Energia Industrial",
    page_icon="⚡",
    layout="wide"
)

# 2. Geração do Dataset Simulado (com pelo menos 20 registros)
@st.cache_data
def carregar_dados():
    np.random.seed(42)
    n_registros = 100  # Gerando 100 registros para um bom treino
    
    velocidade = np.random.uniform(10.0, 60.0, n_registros)
    maquinas = np.random.randint(1, 15, n_registros)
    temp_ambiente = np.random.uniform(18.0, 38.0, n_registros)
    
    # Criando uma relação lógica para o Consumo_kWh com ruído
    consumo = (
        12.5 * velocidade + 
        45.0 * maquinas + 
        3.2 * temp_ambiente + 
        np.random.normal(0, 25, n_registros)
    )
    
    df = pd.DataFrame({
        'Velocidade_Esteira': velocidade,
        'Maquinas_Ativas': maquinas,
        'Temp_Ambiente': temp_ambiente,
        'Consumo_kWh': consumo
    })
    return df

df = carregar_dados()

# 3. Treinamento do Modelo de Regressão Linear
X = df[['Velocidade_Esteira', 'Maquinas_Ativas', 'Temp_Ambiente']]
y = df['Consumo_kWh']

modelo = LinearRegression()
modelo.fit(X, y)

# 4. Painel Lateral (Sidebar) para Entrada de Dados
st.sidebar.header("⚙️ Parâmetros de Entrada")
st.sidebar.markdown("Insira os dados operacionais atuais:")

vel_input = st.sidebar.number_input(
    "Velocidade da Esteira (m/min)", 
    min_value=float(X['Velocidade_Esteira'].min()), 
    max_value=float(X['Velocize_max' if 'Velocize_max' in locals() else 'Velocidade_Esteira'].max() + 10), 
    value=float(X['Velocidade_Esteira'].mean())
)

maq_input = st.sidebar.number_input(
    "Máquinas Ativas", 
    min_value=int(X['Maquinas_Ativas'].min()), 
    max_value=int(X['Maquinas_Ativas'].max() + 5), 
    value=int(X['Maquinas_Ativas'].mean())
)

temp_input = st.sidebar.number_input(
    "Temperatura Ambiente (°C)", 
    min_value=float(X['Temp_Ambiente'].min()), 
    max_value=float(X['Temp_Ambiente'].max() + 10), 
    value=float(X['Temp_Ambiente'].mean())
)

# 5. Tela Principal
st.title("🏭 Painel de Previsão de Consumo de Energia Industrial")
st.markdown("Este painel utiliza Machine Learning (**Regressão Linear**) para prever o consumo energético com base em variáveis operacionais.")

# Preparando os dados inseridos para a previsão
entrada_usuario = pd.DataFrame([[vel_input, maq_input, temp_input]], columns=X.columns)
consumo_previsto = modelo.predict(entrada_usuario)[0]

# Exibindo o consumo previsto em destaque (Métrica)
st.subheader("📊 Resultado da Previsão")
st.metric(
    label="Consumo de Energia Estimado", 
    value=f"{consumo_previsto:.2f} kWh"
)

st.divider()

# 6. Gráfico de Contribuição das Variáveis (Coeficientes do Modelo)
st.subheader("📈 Contribuição de Cada Variável no Modelo")
st.markdown("O gráfico abaixo mostra o peso (coeficiente) de cada parâmetro na predição do consumo:")

fig, ax = plt.subplots(figsize=(8, 4))
coeficientes = modelo.coef_
variaveis = ['Velocidade Esteira', 'Máquinas Ativas', 'Temp. Ambiente']
cores = ['#1f77b4', '#ff7f0e', '#2ca02c']

barras = ax.bar(variaveis, coeficientes, color=cores)
ax.set_ylabel("Impacto no Consumo (kWh por unidade)")
ax.set_title("Coeficientes da Regressão Linear")

# Adicionando os valores em cima das barras
for barra in barras:
    altura = barra.get_height()
    ax.annotate(f'{altura:.2f}',
                xy=(barra.get_x() + barra.get_width() / 2, altura),
                xytext=(0, 3),  # 3 pontos de deslocamento vertical
                textcoords="offset points",
                ha='center', va='bottom')

st.pyplot(fig)

# Expansor opcional para visualizar o dataset gerado
with st.expander("🔍 Visualizar Dataset de Treinamento (Amostra)"):
    st.dataframe(df.head(20))
