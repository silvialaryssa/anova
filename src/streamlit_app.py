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
uploaded_file = 'AmesHousing.csv'  # Default file for demonstration
@st.cache_data
def carregar_dados(uploaded_file):
 return pd.read_csv(uploaded_file)

def exibir_colunas_descricao(df):
    descricoes = {
    'Order': 'Identificador de ordem no dataset',
    'PID': 'Identificador único da propriedade',
    'MS SubClass': 'Tipo de construção (código)',
    'MS Zoning': 'Classificação de zoneamento da propriedade',
    'Lot Frontage': 'Frente do lote (em pés)',
    'Lot Area': 'Área total do lote (em pés quadrados)',
    'Street': 'Tipo de rua de acesso',
    'Alley': 'Tipo de beco de acesso (se houver)',
    'Lot Shape': 'Formato do lote',
    'Land Contour': 'Contorno do terreno',
    'Utilities': 'Serviços públicos disponíveis',
    'Lot Config': 'Configuração do lote',
    'Land Slope': 'Inclinação do terreno',
    'Neighborhood': 'Bairro onde a casa está localizada',
    'Condition 1': 'Proximidade com vias principais ou outras condições',
    'Condition 2': 'Condição adicional',
    'Bldg Type': 'Tipo de edificação',
    'House Style': 'Estilo da residência',
    'Overall Qual': 'Qualidade geral do material e acabamento',
    'Overall Cond': 'Condição geral da casa',
    'Year Built': 'Ano de construção',
    'Year Remod/Add': 'Ano da última reforma ou adição',
    'Roof Style': 'Estilo do telhado',
    'Roof Matl': 'Material do telhado',
    'Exterior 1st': 'Acabamento externo primário',
    'Exterior 2nd': 'Acabamento externo secundário',
    'Mas Vnr Type': 'Tipo de revestimento de alvenaria',
    'Mas Vnr Area': 'Área de revestimento de alvenaria',
    'Exter Qual': 'Qualidade do acabamento externo',
    'Exter Cond': 'Condição do acabamento externo',
    'Foundation': 'Tipo de fundação',
    'Bsmt Qual': 'Qualidade do porão',
    'Bsmt Cond': 'Condição do porão',
    'Bsmt Exposure': 'Exposição do porão à luz natural',
    'BsmtFin Type 1': 'Tipo de acabamento do porão 1',
    'BsmtFin SF 1': 'Área do porão finalizada (tipo 1)',
    'BsmtFin Type 2': 'Tipo de acabamento do porão 2',
    'BsmtFin SF 2': 'Área do porão finalizada (tipo 2)',
    'Bsmt Unf SF': 'Área do porão não finalizada',
    'Total Bsmt SF': 'Área total do porão',
    'Heating': 'Tipo de aquecimento',
    'Heating QC': 'Qualidade do sistema de aquecimento',
    'Central Air': 'Possui ar condicionado central',
    'Electrical': 'Sistema elétrico',
    '1st Flr SF': 'Área do primeiro andar',
    '2nd Flr SF': 'Área do segundo andar',
    'Low Qual Fin SF': 'Área de baixa qualidade finalizada',
    'Gr Liv Area': 'Área total habitável acima do solo',
    'Bsmt Full Bath': 'Banheiro completo no porão',
    'Bsmt Half Bath': 'Meio banheiro no porão',
    'Full Bath': 'Banheiros completos acima do solo',
    'Half Bath': 'Meios banheiros acima do solo',
    'Bedroom AbvGr': 'Número de quartos acima do solo',
    'Kitchen AbvGr': 'Número de cozinhas acima do solo',
    'Kitchen Qual': 'Qualidade da cozinha',
    'TotRms AbvGrd': 'Total de cômodos acima do solo',
    'Functional': 'Funcionalidade da casa',
    'Fireplaces': 'Número de lareiras',
    'Fireplace Qu': 'Qualidade das lareiras',
    'Garage Type': 'Tipo de garagem',
    'Garage Yr Blt': 'Ano de construção da garagem',
    'Garage Finish': 'Acabamento da garagem',
    'Garage Cars': 'Capacidade de carros na garagem',
    'Garage Area': 'Área da garagem',
    'Garage Qual': 'Qualidade da garagem',
    'Garage Cond': 'Condição da garagem',
    'Paved Drive': 'Entrada pavimentada',
    'Wood Deck SF': 'Área do deck de madeira',
    'Open Porch SF': 'Área da varanda aberta',
    'Enclosed Porch': 'Área da varanda fechada',
    '3Ssn Porch': 'Área da varanda de três estações',
    'Screen Porch': 'Área da varanda com tela',
    'Pool Area': 'Área da piscina',
    'Pool QC': 'Qualidade da piscina',
    'Fence': 'Tipo de cerca',
    'Misc Feature': 'Recursos adicionais (elevador, etc.)',
    'Misc Val': 'Valor dos recursos adicionais',
    'Mo Sold': 'Mês da venda',
    'Yr Sold': 'Ano da venda',
    'Sale Type': 'Tipo de venda',
    'Sale Condition': 'Condição da venda',
    'SalePrice': 'Preço final de venda da casa'
}


    df.columns = df.columns.str.replace(' ', '_')

    st.subheader("Descrição das Colunas")
    colunas_df = pd.DataFrame({
        "Coluna": df.columns,
        "Descrição": [descricoes.get(col.replace('_', ' '), descricoes.get(col, "")) for col in df.columns]
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

    p_shapiro = shapiro.pvalue
    st.write(f"Shapiro-Wilk (Normalidade dos resíduos): {p_shapiro:.4f}")
    if p_shapiro >= 0.05:
        st.success("✅ Os resíduos seguem uma distribuição normal (p ≥ 0.05).")
    else:
        st.warning("⚠️ Os resíduos **não seguem** uma distribuição normal (p < 0.05).")

    p_bp = bp_test[1]
    st.write(f"Breusch-Pagan (Homocedasticidade dos resíduos): {p_bp:.4f}")
    if p_bp >= 0.05:
        st.success("✅ Variância constante dos resíduos (homocedasticidade verificada).")
    else:
        st.warning("⚠️ Os resíduos **não têm variância constante** (heterocedasticidade detectada).")

    if shapiro.pvalue < 0.05 or bp_test[1] < 0.05:
        kruskal = stats.kruskal(*grupos)
        st.warning(f"ANOVA não atende pressupostos. Usando Kruskal-Wallis: p = {kruskal.pvalue:.4f}")
    else:
        st.success("Pressupostos atendidos para ANOVA tradicional")

# ================================
# ENTRADA DE DADOS
# ================================
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
# Boxplots com Altair
# ================================
st.header("Boxplots para Visualização das Variáveis")
for var in [var1, var2, var3]:
    chart_data = df_clean[[var, var_target]].dropna()
    chart = alt.Chart(chart_data).mark_boxplot(extent='min-max').encode(
        x=alt.X(f'{var}:N', title=var),
        y=alt.Y(f'{var_target}:Q', title='Preço de Venda'),
        color=alt.Color(f'{var}:N', legend=None)
    ).properties(width=400, height=200)
    st.altair_chart(chart, use_container_width=True)

# ================================
# AVALIAÇÃO DAS VARIÁVEIS
# ================================
st.header("Avaliação Estatística das Variáveis")
for var in [var1, var2, var3]:
    avaliar_variavel(var, df_clean, var_target)

# ================================
# POST-HOC: Teste de Tukey
# ================================
def tukey_posthoc_plot(df, var_cat, var_target):
    st.subheader(f"Teste Post-Hoc: Tukey HSD para {var_cat}")
    st.subheader(f"Tukey HSD: Comparações entre categorias de {var_cat}")
    try:
        tukey = pairwise_tukeyhsd(endog=df[var_target], groups=df[var_cat], alpha=0.05)
        tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])

        # Conversão rigorosa para booleano
        tukey_df['reject'] = tukey_df['reject'].apply(lambda x: str(x).strip().lower() == 'true')
        # DataFrame final completo
        st.write(f"Total de comparações: {len(tukey_df)}")
        tukey_df

        tukey_df['meandiff'] = pd.to_numeric(tukey_df['meandiff'], errors='coerce')
        tukey_df['p-adj'] = pd.to_numeric(tukey_df['p-adj'], errors='coerce')

        # Filtrando apenas as comparações significativas
  
        st.write(f"Comparações significativas (p < 0.05): {tukey_df[tukey_df['reject']].shape[0]}")
        sig_df = tukey_df[tukey_df['reject'] == True].copy()
        
        

        if sig_df.empty:
            st.info("Nenhuma diferença estatística significativa encontrada entre os pares de categorias.")
        else:
            sig_df['Comparison'] = sig_df['group1'].astype(str) + ' vs ' + sig_df['group2'].astype(str)
            sig_df['meandiff'] = sig_df['meandiff'].astype(float)

            st.dataframe(sig_df[['Comparison', 'meandiff', 'p-adj', 'reject']])

            chart = alt.Chart(sig_df).mark_bar(color='orange').encode(
                x=alt.X('meandiff:Q', title='Diferença de médias'),
                y=alt.Y('Comparison:N', sort='-x', title='Comparação'),
                tooltip=['Comparison', 'meandiff', 'p-adj']
            ).properties(width=400, height=200)

            st.subheader(f"Gráfico de Diferenças de Médias - {var_cat}")
            st.altair_chart(chart, use_container_width=True)

            # Interpretação automática
            n = len(sig_df)
            maiores_diffs = sig_df.loc[sig_df['meandiff'].abs().nlargest(3).index]
            exemplo = maiores_diffs.iloc[0]
            interpretacao = (
                f"💡 Foram encontradas **{n} comparações com diferenças significativas** entre as categorias de **{var_cat}**.\n\n"
                f"A maior diferença foi observada entre **{exemplo['Comparison']}**, com uma média de diferença de aproximadamente "
                f"**{exemplo['meandiff']:.2f}** no preço de venda.\n\n"
                f"Essas diferenças indicam que algumas categorias de **{var_cat}** influenciam significativamente os preços médios das casas."
            )
            st.markdown(interpretacao)
    except Exception as e:
        st.warning(f"Erro ao executar Tukey para {var_cat}: {e}")

for var in [var1, var2, var3]:
    tukey_posthoc_plot(df_clean, var, var_target)


# ================================
# RELATÓRIO FINAL
# ================================
st.header("📘 Relatório Final - Análise de Variância (ANOVA) no Ames Housing Dataset")

st.markdown("""
### 🔍 1. Análise dos Q-Q Plots
Os Q-Q Plots das médias de preço por categoria foram utilizados para verificar a normalidade das médias dos grupos para cada variável categórica analisada:

**Neighborhood (Bairro):**  
Os pontos se afastam significativamente da linha de referência, sugerindo violação da normalidade — o que indica que os bairros possuem padrões distintos e não seguem uma distribuição normal conjunta.

**House_Style (Estilo da Casa):**  
Os pontos estão relativamente próximos da linha, com leves desvios — o que sugere uma distribuição aproximadamente normal, embora outros testes sejam necessários para confirmar.

**Bsmt_Full_Bath (Banheiro Completo no Porão):**  
Os pontos estão muito próximos da linha reta, indicando uma forte aderência à normalidade das médias entre os grupos.

---

### 📦 2. Análise dos Boxplots
Os boxplots mostram a distribuição do preço de venda para cada categoria das variáveis:

**Neighborhood:**  
Apresenta grande variabilidade nos preços dentro e entre os bairros, com dispersões heterogêneas e valores discrepantes, reforçando a ideia de diferenças significativas entre os grupos.

**House_Style:**  
As distribuições são mais homogêneas, mas ainda há diferenças visíveis nas medianas, especialmente entre estilos mais comuns e menos comuns.

**Bsmt_Full_Bath:**  
Os grupos têm uma distribuição mais clara, com aumento progressivo dos preços com o número de banheiros, sugerindo uma tendência linear ou ordinal.

---

### 📈 3. Testes Estatísticos (ANOVA e Pós-Hoc)

| Variável          | ANOVA p-valor | Shapiro-Wilk (Normalidade) | Breusch-Pagan (Homoscedasticidade) | ANOVA Tradicional Adequada? |
|-------------------|---------------|-----------------------------|------------------------------------|-----------------------------|
| Neighborhood      | < 0.0001      | ❌ Não (p < 0.05)           | ❌ Não (p < 0.05)                   | ❌ Não                      |
| House_Style       | < 0.0001      | ❌ Não (p < 0.05)           | ❌ Não (p < 0.05)                   | ❌ Não                      |
| Bsmt_Full_Bath    | < 0.0001      | ❌ Não (p < 0.05)           | ❌ Não (p < 0.05)                   | ❌ Não                      |

📌 **Conclusão:** Em nenhuma das variáveis os pressupostos da ANOVA tradicional foram atendidos. Portanto, testes alternativos não paramétricos foram utilizados.

---

### 🔁 Kruskal-Wallis
Todas as variáveis apresentaram **p-valor < 0.05**, confirmando que há **diferenças estatísticas significativas entre os grupos** em cada uma delas.

---

### 🔬 Teste Post Hoc (Tukey HSD)
O teste de Tukey HSD identificou várias diferenças significativas entre pares de categorias para todas as variáveis analisadas.  
As comparações com maiores diferenças de médias foram evidenciadas nos gráficos e tabelas geradas no app.  
O gráfico de barras auxilia na interpretação visual dos pares com diferenças mais relevantes.

---

### 🧠 Conclusão Geral
As variáveis categóricas **Neighborhood**, **House_Style** e **Bsmt_Full_Bath** influenciam significativamente o preço de venda das casas.  
A ANOVA tradicional não foi adequada, pois os testes de normalidade e homocedasticidade falharam para todas as variáveis.  
O uso de testes **não paramétricos** como o **Kruskal-Wallis** foi essencial e revelou diferenças significativas entre os grupos.  
O teste de **Tukey HSD** complementou a análise, detalhando quais pares de categorias apresentam as maiores diferenças de preço.


### 🧠 Conclusão Geral
**PPCA**: Programa de Computação Aplicada - UNB  
**AEDI**: Análise Estatística de Dados e Informações  
**Prof.** João Gabriel de Moraes Souza
**Aluna:** Silva Laryssa Branco da Silva
**Data:** 2024-01-15

---


### Autores e Referências
- **PPCA**: Programa de Computação Aplicada - UNB  
- **AEDI**: Análise Estatística de Dados e Informações  
- **Prof.** João Gabriel de Moraes Souza  
- **Aluna**: Silva Laryssa Branco da Silva  
- **Data**: 2024-01-15


### 🔗 Links

- 📊 Projeto no Community Cloud: [https://aedianova.streamlit.app/](https://aedianova.streamlit.app/)  
- 💻 Código fonte GitHub: [https://github.com/silvialaryssa/anova](https://github.com/silvialaryssa/anova)


""")


