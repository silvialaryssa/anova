# anovapp.py - Aplicativo modularizado de análise ANOVA

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import altair as alt


# ================================
# CONFIGURAÇÕES INICIAIS
# ================================
st.set_page_config(page_title="Análise ANOVA - Ames Housing", layout="wide")
st.title("Análise Estatística com ANOVA - Ames Housing Dataset")

# ================================
# FUNÇÕES UTILITÁRIAS
# ================================
uploaded_file = 'src/AmesHousing.csv'  # Default file for demonstration
@st.cache_data
def carregar_dados(uploaded_file):
 return pd.read_csv(uploaded_file)

def exibir_colunas_descricao(df):
    descricoes = {        
        'Order': 'Identificador de ordem no dataset',
        'PID': 'Identificador único da propriedade',
        'MS_SubClass': 'Tipo de construção (código)',
        'Neighborhood': 'Bairro onde a casa está localizada',
        'House_Style': 'Estilo da residência',
        'Bsmt_Full_Bath': 'Banheiro completo no porão',
        'SalePrice': 'Preço final de venda da casa'
        # Adicione outras descrições conforme necessário}  # abreviado por brevidade
    }
    
    # Garante que os nomes das colunas estejam sem espaços
    df.columns = df.columns.str.replace(' ', '_')
    

    # Cria DataFrame com colunas e descrições
    st.subheader("Descrição das Colunas")
    colunas_df = pd.DataFrame({
        "Coluna": df.columns,
        "Descrição": [descricoes.get(col, "") for col in df.columns]
    })
    st.dataframe(colunas_df)

def qq_plot_medias(df, var_categ, var_target):
    medias = df.groupby(var_categ)[var_target].mean().dropna()
    fig = plt.figure(figsize=(4, 3))
    stats.probplot(medias, dist="norm", plot=plt)
    plt.title(f"Q-Q Plot das Médias de {var_target} por {var_categ}", fontsize=10)
    plt.xlabel("Quantis teóricos", fontsize=8)
    plt.ylabel("Quantis amostrais", fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    return fig

def avaliar_variavel(var, df_clean, var_target):
    st.subheader(f"Variável: {var}")
    grupos = [group[var_target].values for name, group in df_clean.groupby(var)]
    anova = stats.f_oneway(*grupos)
    st.write(f"p-valor da ANOVA: {anova.pvalue:.6f}")
    
        
    # Interpretação automática
    if anova.pvalue < 0.001:
        st.markdown("🔬 **Conclusão**: Existe uma **diferença estatisticamente muito significativa** entre as médias dos grupos.")
    elif anova.pvalue < 0.05:
        st.markdown("🔬 **Conclusão**: Existe uma **diferença estatisticamente significativa** entre as médias dos grupos.")
    else:
        st.markdown("📊 **Conclusão**: **Não há evidência estatística suficiente** para afirmar que as médias dos grupos são diferentes.")
        

    modelo = sm.OLS.from_formula(f'{var_target} ~ C({var})', data=df_clean).fit()
    residuos = modelo.resid
    shapiro = stats.shapiro(residuos)
    bp_test = het_breuschpagan(residuos, modelo.model.exog)

   # st.write(f"Shapiro-Wilk: {shapiro.pvalue:.4f}")
   # st.write(f"Breusch-Pagan: {bp_test[1]:.4f}")
    
        # Shapiro-Wilk (normalidade dos resíduos)
    p_shapiro = shapiro.pvalue
    st.write(f"Shapiro-Wilk (Normalidade dos resíduos): {p_shapiro:.4f}")
    if p_shapiro >= 0.05:
        st.success("✅ Os resíduos seguem uma distribuição normal (p ≥ 0.05).")
    else:
        st.warning("⚠️ Os resíduos **não seguem** uma distribuição normal (p < 0.05).")

    # Breusch-Pagan (homocedasticidade dos resíduos)
    p_bp = bp_test[1]
    st.write(f"Breusch-Pagan (Homocedasticidade dos resíduos): {p_bp:.4f}")
    if p_bp >= 0.05:
        st.success("✅ Variância constante dos resíduos (homocedasticidade verificada).")
    else:
        st.warning("⚠️ Os resíduos **não têm variância constante** (heterocedasticidade detectada).")
    
    
    

    if shapiro.pvalue < 0.05 or bp_test[1] < 0.05:
        kruskal = stats.kruskal(*grupos)
        st.warning(f"Teste de Kruskal-Wallis: p = {kruskal.pvalue:.4f}")
    else:
        st.success("Pressupostos atendidos para ANOVA tradicional")

    try:
        tukey = pairwise_tukeyhsd(endog=df_clean[var_target], groups=df_clean[var], alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
        st.dataframe(tukey_df)
    except Exception as e:
        st.warning(f"Erro no Tukey HSD: {e}")

# ================================
# ENTRADA DE DADOS
# ================================
#uploaded_file = st.file_uploader("📁 Envie o arquivo AmesHousing.csv", type=["csv"])
#if not uploaded_file:
#    st.stop()

df = carregar_dados(uploaded_file)
df.columns = df.columns.str.replace(' ', '_')
st.success("Arquivo carregado com sucesso!")
exibir_colunas_descricao(df)

# ================================
# DEFINIÇÃO DE VARIÁVEIS
# ================================
var_target = 'SalePrice'
var1 = 'Neighborhood'
var2 = 'House_Style'
var3 = 'Bsmt_Full_Bath'
df_clean = df[[var_target, var1, var2, var3]].dropna()

# ================================
# Q-Q Plots
# ================================
st.header("Q-Q Plot das Médias por Variável")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(var1)
    st.pyplot(qq_plot_medias(df_clean, var1, var_target))
with col2:
    st.subheader(var2)
    st.pyplot(qq_plot_medias(df_clean, var2, var_target))
with col3:
    st.subheader(var3)
    st.pyplot(qq_plot_medias(df_clean, var3, var_target))

# ================================
# ANÁLISE EXPLORATÓRIA
# ================================
st.header("Boxplots para Visualização das Variáveis")
for var in [var1, var2, var3]:
    chart_data = df_clean[[var, var_target]].dropna()
    chart = alt.Chart(chart_data).mark_boxplot(extent='min-max').encode(
        x=alt.X(f'{var}:N', title=var),
        y=alt.Y(f'{var_target}:Q', title='Preço de Venda'),
        color=alt.Color(f'{var}:N', legend=None)
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)

# ================================
# AVALIAÇÃO DAS VARIÁVEIS
# ================================
st.header("Avaliação Estatística das Variáveis")
for var in [var1, var2, var3]:
    avaliar_variavel(var, df_clean, var_target)

# ================================
# EXPORTÁVEL COMO MÓDULO
# ================================
# Para usar como biblioteca:
# from anovapp import avaliar_variavel, qq_plot_medias
# e executar manualmente com df desejado

# ================================
# AVISO FINAL
# ================================
st.warning("Por favor, envie o dataset CSV para começar a análise.")


# ================================
# Interpretação do Resultado
# ================================
st.header("Interpretação dos Q-Q Plots das Médias por Variável")
st.markdown("""
**1. Neighborhood (Bairro)**  
**Interpretação**: Os pontos se afastam da linha reta em ambas as extremidades, o que indica desvio da normalidade (provavelmente caudas mais pesadas ou assimetria).  
**Conclusão**: A distribuição das médias de preço entre os bairros não segue bem uma distribuição normal, sugerindo que os bairros têm efeitos bem distintos no preço.

**2. House_Style (Estilo da Casa)**  
**Interpretação**: Os pontos estão razoavelmente próximos da linha vermelha, com pequenas variações.  
**Conclusão**: A distribuição das médias de preço por estilo de casa é aproximadamente normal. A ANOVA tradicional pode ser mais adequada aqui.

**3. Bsmt_Full_Bath (Banheiro Completo no Porão)**  
**Interpretação**: Apesar do número pequeno de grupos, os pontos estão muito próximos da linha reta, indicando forte aderência à normalidade.  
**Conclusão**: A variável Bsmt_Full_Bath apresenta uma distribuição de médias normal — é uma variável promissora para análise com testes paramétricos como ANOVA.
""")


st.header("📄 Interpretação Manual dos Testes de Hipóteses")

with st.expander("🔍 Análise Detalhada de Cada Variável"):
    st.markdown("""
### 1. Variável: `Neighborhood`
- **p-valor da ANOVA**: `0.000000`
- **Shapiro-Wilk (normalidade dos resíduos)**: `0.0000`
- **Breusch-Pagan (homocedasticidade)**: `0.0000`
- **Kruskal-Wallis**: `p = 0.0000`

**🧠 Conclusão**:  
A ANOVA tradicional não é adequada, pois os resíduos **não seguem distribuição normal** (p < 0.05) e **não apresentam homocedasticidade** (p < 0.05).  
Como alternativa, utilizamos o **teste de Kruskal-Wallis**, que **não depende desses pressupostos**.  
📌 O resultado (**p = 0.0000**) indica que **existe diferença estatisticamente significativa entre os bairros** em relação ao preço de venda.

---

### 2. Variável: `House_Style`
- **p-valor da ANOVA**: `0.000000`
- **Shapiro-Wilk (normalidade dos resíduos)**: `0.0000`
- **Breusch-Pagan (homocedasticidade)**: `0.0000`
- **Kruskal-Wallis**: `p = 0.0000`

**🧠 Conclusão**:  
A ANOVA tradicional não é apropriada, pois os resíduos **não são normais** e há **variâncias diferentes entre os grupos**.  
📌 O **teste Kruskal-Wallis** foi aplicado como alternativa robusta.  
O **p-valor significativo (p = 0.0000)** indica que o **estilo da casa influencia significativamente o preço de venda**.

---

### 3. Variável: `Bsmt_Full_Bath`
- **p-valor da ANOVA**: `0.000000`
- **Shapiro-Wilk (normalidade dos resíduos)**: `0.0000`
- **Breusch-Pagan (homocedasticidade)**: `0.0000`
- **Kruskal-Wallis**: `p = 0.0000`

**🧠 Conclusão**:  
Novamente, os pressupostos da ANOVA foram violados.  
Como os resíduos **não são normais** e **não há homogeneidade de variância**, utilizou-se o **teste de Kruskal-Wallis**, que **confirmou diferenças estatisticamente significativas (p = 0.0000)** entre os grupos de número de banheiros completos no porão.

---
### ✅ Resumo Final
Para **todas as variáveis analisadas**, a ANOVA tradicional **não foi adequada** devido à violação dos pressupostos de **normalidade** e **homocedasticidade**.  
O **teste Kruskal-Wallis**, que é mais **robusto** em cenários como este, foi utilizado com sucesso e revelou **diferenças significativas entre os grupos** em todas as variáveis.  
🔍 Isso **indica que cada uma dessas variáveis influencia significativamente o preço de venda** das residências.
""")
