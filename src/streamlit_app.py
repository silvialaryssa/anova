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
        'MS_SubClass': 'Tipo de construção (código)',
        'Neighborhood': 'Bairro onde a casa está localizada',
        'House_Style': 'Estilo da residência',
        'Bsmt_Full_Bath': 'Banheiro completo no porão',
        'SalePrice': 'Preço final de venda da casa'
    }

    df.columns = df.columns.str.replace(' ', '_')

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
st.header("📄 Relatório Final da Análise")
st.markdown("""
### Análise dos Q-Q Plots
- **Neighborhood**: desvio da normalidade → efeitos distintos no preço.
- **House_Style**: aproximação razoável à normalidade.
- **Bsmt_Full_Bath**: boa aderência à normalidade.

### Análise dos Boxplots
- **Neighborhood**: variações amplas e heterogêneas.
- **House_Style**: diferenças visíveis nas medianas.
- **Bsmt_Full_Bath**: tendência clara de aumento de preço com número de banheiros.

### Conclusões dos Testes Estatísticos
- **Todos os p-valores da ANOVA** foram menores que 0.001.
- **Todos os testes de Shapiro-Wilk e Breusch-Pagan** indicaram violação dos pressupostos.
- Utilizado **teste de Kruskal-Wallis** como alternativa.

### Testes Post Hoc (Tukey HSD)
- Diferenciação estatística significativa entre diversas categorias.
- Evidências claras de influência dessas variáveis no **preço de venda**.

### Conclusão Geral
A ANOVA tradicional não foi adequada devido à violação dos pressupostos de normalidade e homocedasticidade. O uso do teste **Kruskal-Wallis** foi necessário e apropriado. Com base nas análises, conclui-se que as variáveis **Neighborhood**, **House_Style** e **Bsmt_Full_Bath** têm **influência estatisticamente significativa sobre os preços das casas** no dataset Ames Housing.
""")
