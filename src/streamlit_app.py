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

    st.subheader("Descrição de todas as colunas")
    colunas_df = pd.DataFrame({
        "Coluna": df.columns,
        "Descrição": [descricoes.get(col.replace('_', ' '), descricoes.get(col, "")) for col in df.columns]
    })
    st.dataframe(colunas_df)
    
# Categorias selecionadas para análise
    st.subheader("Colunas selecionadas para análise")
    colunas_selecionadas = ['SalePrice', 'Neighborhood', 'House_Style', 'Bsmt_Full_Bath']    
    st.dataframe(df[colunas_selecionadas].head())
    
# ================================
# Q-Q Plot das Médias    
# ================================    

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

# ================================
# ANOVA E AVALIAÇÃO DAS VARIÁVEIS
# ================================

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
        st.success("Pressupostos não atendidos, logo o teste não paramétrico - Kruskal-Wallis foi aplicado.")
        if kruskal.pvalue < 0.001:
            st.markdown("🔬 **Conclusão**: Existe uma **diferença estatisticamente muito significativa** entre as medianas dos grupos.")
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
#var3 = 'Fence'  # Alterado para 'Yr_Sold' como exemplo
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
# ANOVA de múltiplos fatores (Two-Way ou mais)
# ================================    
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Modelo ANOVA com 3 variáveis categóricas
##modelo = smf.ols('SalePrice ~ C(Neighborhood) + C(House_Style) + C(Bsmt_Full_Bath)', data=df_clean).fit()
##anova_tabela = sm.stats.anova_lm(modelo, typ=2)

##st.header("🧠 Interpretação dos Resultados - ANOVA - Tow Way")

##for fator in ['C(Neighborhood)', 'C(House_Style)', 'C(Bsmt_Full_Bath)']:
##    p_valor = anova_tabela.loc[fator, 'PR(>F)']
##    f_stat  = anova_tabela.loc[fator, 'F']
    
##    if p_valor < 0.001:
##        st.success(f"🔹 {fator}: Influência **muito significativa** sobre o preço de venda (F = {f_stat:.2f}, p < 0.001).")
##    elif p_valor < 0.05:
##        st.info(f"🔹 {fator}: Influência **significativa** sobre o preço de venda (F = {f_stat:.2f}, p = {p_valor:.4f}).")
##    else:
##        st.warning(f"🔹 {fator}: **Sem influência estatisticamente significativa** (F = {f_stat:.2f}, p = {p_valor:.4f}).")



##st.subheader("📊 ANOVA Two-way (Neighborhood + House_Style + Bsmt_Full_Bath)")
##st.dataframe(anova_tabela)

#modelo_interacao = smf.ols('SalePrice ~ C(Neighborhood) * C(House_Style) + C(Bsmt_Full_Bath)', data=df_clean).fit()
#anova_inter = sm.stats.anova_lm(modelo_interacao, typ=2)
#st.subheader("📊 ANOVA com Interação (Neighborhood * House_Style)")
#st.dataframe(anova_inter)

#######################################################################################################################
import statsmodels.formula.api as smf
import statsmodels.api as sm

def anova_multifatorial(df, var_target, fatores):
    """
    Executa ANOVA multifatorial e interpreta resultados.

    Parâmetros:
    - df: DataFrame com os dados
    - var_target: string com o nome da variável resposta
    - fatores: lista de strings com os nomes das variáveis categóricas
    """

    # Monta a fórmula para o modelo, aplicando C() em cada fator
    fatores_formula = ' + '.join([f'C({f})' for f in fatores])
    formula = f'{var_target} ~ {fatores_formula}'

    # Ajusta o modelo e calcula ANOVA
    modelo = smf.ols(formula, data=df).fit()
    anova_tabela = sm.stats.anova_lm(modelo, typ=2)

    # Título interpretativo
    st.header(f"🧠 Interpretação dos Resultados - ANOVA  Two-way")
    
    # Interpretação de cada fator
    for fator in [f'C({f})' for f in fatores]:
        p_valor = anova_tabela.loc[fator, 'PR(>F)']
        f_stat  = anova_tabela.loc[fator, 'F']
        
        if p_valor < 0.001:
            st.success(f"🔹 {fator}: Influência **muito significativa** (F = {f_stat:.2f}, p < 0.001).")
        elif p_valor < 0.05:
            st.info(f"🔹 {fator}: Influência **significativa** (F = {f_stat:.2f}, p = {p_valor:.4f}).")
        else:
            st.warning(f"🔹 {fator}: **Sem influência significativa** (F = {f_stat:.2f}, p = {p_valor:.4f}).")

    # Exibe a tabela ANOVA
    st.subheader(f"📊 ANOVA Two-way ({' + '.join(fatores)})")
    st.dataframe(anova_tabela)

    return anova_tabela  # opcional: retorna a tabela para uso externo

fatores = [var1,var2,var3]
anova_multifatorial(df_clean, var_target='SalePrice', fatores=fatores)




######################################################################################################################

       
# ================================
# AVALIAÇÃO DAS VARIÁVEIS
# ================================
st.header("🧠 Interpretação dos Resultados - ANOVA On way para cada Variável")
for var in [var1, var2, var3]:
    avaliar_variavel(var, df_clean, var_target)

# ================================
# POST-HOC: Teste de Tukey
# ================================
def tukey_posthoc_plot(df, var_cat, var_target):
    st.subheader(f"Teste Post-Hoc: Tukey HSD - Para sabe onde é a dirença dentro do gurpo {var_cat}")
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

#for var in [var1, var2, var3]:
    #tukey_posthoc_plot(df_clean, var, var_target)


# Gameshowe's test
import pingouin as pg
import altair as alt

def gameshowell_posthoc_plot(df, var_cat, var_target):
    st.subheader(f"Teste Post-Hoc: Games-Howell - Comparações em {var_cat}")

    try:
        # Garantindo os tipos corretos
        df[var_cat] = df[var_cat].astype(str)  # trata como categórica
        df[var_target] = pd.to_numeric(df[var_target], errors='coerce')
        dados = df[[var_cat, var_target]].dropna()

        # Aplicando o teste de Games-Howell
        resultado = pg.pairwise_gameshowell(dv=var_target, between=var_cat, data=dados)

        # Filtro de comparações significativas
        resultado['significant'] = resultado['pval'] < 0.05
        sig_df = resultado[resultado['significant']].copy()

        st.write(f"Total de comparações: {len(resultado)}")
        st.write(f"Comparações significativas (p < 0.05): {len(sig_df)}")

        if sig_df.empty:
            st.info("Nenhuma diferença estatística significativa encontrada entre os pares de categorias.")
            return

        # Preparando os dados para visualização
        sig_df['Comparison'] = sig_df['A'] + ' vs ' + sig_df['B']
        sig_df['meandiff'] = sig_df['diff']

        # Tabela com os principais dados
        st.dataframe(sig_df[['Comparison', 'meandiff', 'pval', 'significant']])

        # Gráfico de barras
        chart = alt.Chart(sig_df).mark_bar(color='orange').encode(
            x=alt.X('meandiff:Q', title='Diferença de Médias'),
            y=alt.Y('Comparison:N', sort='-x', title='Comparação'),
            tooltip=['Comparison', 'meandiff', 'pval']
        ).properties(width=400, height=250)

        st.subheader(f"Gráfico de Diferenças de Médias - {var_cat}")
        st.altair_chart(chart, use_container_width=True)

        # Interpretação automática
        maiores_diffs = sig_df.loc[sig_df['meandiff'].abs().nlargest(3).index]
        exemplo = maiores_diffs.iloc[0]
        interpretacao = (
            f"💡 Foram encontradas **{len(sig_df)} comparações com diferenças significativas** entre as categorias de **{var_cat}**.\n\n"
            f"A maior diferença foi observada entre **{exemplo['Comparison']}**, com uma média de diferença de aproximadamente "
            f"**{exemplo['meandiff']:.2f}** no preço de venda.\n\n"
            f"Essas diferenças indicam que algumas categorias de **{var_cat}** influenciam significativamente os preços médios das casas."
        )
        st.markdown(interpretacao)

    except Exception as e:
        st.error(f"Erro ao executar Games-Howell para {var_cat}: {e}")
# Executando o Games-Howell para cada variável categórica
for var in [var1, var2, var3]:
    gameshowell_posthoc_plot(df_clean, var, var_target)




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

### 📊 4. Teste de Shapiro-Wilk

O teste de Shapiro-Wilk verifica se uma distribuição é significativamente diferente  
de uma normal. Embora eficaz, ele é sensível a grandes amostras, nas quais  
pequenos desvios da normalidade já geram p-valores baixos.

No nosso caso, foi usado para testar a **normalidade dos resíduos da ANOVA**.  
Todas as variáveis apresentaram **p < 0.05**, indicando violação da normalidade.

---

### 🔁 Teste Não Paramétrico (Kruskal-Wallis)

Segundo Andy Field (2009), a ANOVA de um fator tem como equivalente não paramétrico  
o **teste de Kruskal-Wallis**, recomendado quando pressupostos como normalidade  
ou homocedasticidade são violados.

Diante da violação dos pressupostos, aplicamos o Kruskal-Wallis.  
**Todas as variáveis apresentaram p < 0.05**, confirmando diferenças entre os grupos.

---

### 🔬 Teste Post Hoc (Games-Howell)

Andy Field (2009) recomenda o **teste de Games-Howell** quando há dúvida sobre  
a homogeneidade das variâncias ou quando os tamanhos amostrais são muito diferentes.  
É uma alternativa robusta ao teste de Tukey tradicional.

Substituímos o Tukey pelo Games-Howell, que identificou  
**diferenças estatísticas significativas entre as categorias** para todas as variáveis.

---

### 🧠 Conclusão Geral

As variáveis **Neighborhood**, **House_Style** e **Bsmt_Full_Bath** afetam de forma  
estatisticamente significativa o preço de venda das casas.  

Como os pressupostos da ANOVA tradicional foram violados, utilizamos o **Kruskal-Wallis**,  
e como teste post hoc, o **Games-Howell**, apropriado para variâncias desiguais.  
Ambos os testes reforçaram a presença de diferenças relevantes entre os grupos.

---

### 📚 Referências
- Field, A. (2009). Descobrindo a estatística usando o SPSS. 2. ed. Porto Alegre: Artmed, 2009

---

### Autores
- **PPCA**: Programa de Computação Aplicada - UNB  
- **AEDI**: Análise Estatística de Dados e Informações  
- **Prof.** João Gabriel de Moraes Souza  
- **Aluna**: Silva Laryssa Branco da Silva  
- **Data**: 2024-01-15


### 🔗 Links

- 📊 Projeto no Community Cloud: [https://aedianova.streamlit.app/](https://aedianova.streamlit.app/)  
- 💻 Código fonte GitHub: [https://github.com/silvialaryssa/anova](https://github.com/silvialaryssa/anova)


""")


#import pandas as pd
#import statsmodels.formula.api as smf
#from statsmodels.stats.diagnostic import het_breuschpagan
#from scipy import stats
#import streamlit as st

# ========================
# Avaliação dos pressupostos
# ========================


#st.header("🧪 Avaliação dos Pressupostos da ANOVA - Variáveis Categóricas")

#cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remoções seguras (evita erro se não existir)
#for col in ['SalePrice', 'PID']:
#    if col in cat_cols:
#        cat_cols.remove(col)

#resultado_pressupostos = []

#for var in cat_cols:
 #   try:
  #      grupos = [group['SalePrice'].values for name, group in df.groupby(var)]
        
        # Remove variáveis com apenas 1 grupo ou grupos muito pequenos
  #      if len(grupos) < 2 or any(len(g) < 3 for g in grupos):
  #          continue

        # ANOVA
  #      modelo = smf.ols(f'SalePrice ~ C({var})', data=df).fit()
  #      residuos = modelo.resid

        # Testes
  #     shapiro = stats.shapiro(residuos)
  #      bp_test = het_breuschpagan(residuos, modelo.model.exog)

  #      resultado_pressupostos.append({
  #          'Variável': var,
  #          'Grupos': len(grupos),
  #          'Shapiro-Wilk (p)': round(shapiro.pvalue, 4),
  #          'Breusch-Pagan (p)': round(bp_test[1], 4),
  #          'Atende Pressupostos': shapiro.pvalue >= 0.05 and bp_test[1] >= 0.05
  #      })
  #  except Exception as e:
  #      st.warning(f"⚠️ Erro ao processar {var}: {e}")

# ========================
# Exibição dos resultados
# ========================

#if resultado_pressupostos:
#    df_resultado = pd.DataFrame(resultado_pressupostos)

#    st.subheader("📋 Resultado dos Testes de Pressupostos")
#    st.dataframe(df_resultado.style.applymap(
#        lambda val: 'background-color: #d4edda' if val is True else
#                    'background-color: #f8d7da' if val is False else '',
#        subset=['Atende Pressupostos']
#    ))

#    variaveis_validas = df_resultado[df_resultado['Atende Pressupostos']]['Variável'].tolist()
#    st.markdown("✅ **Variáveis que atendem aos pressupostos da ANOVA:**")
#    st.success(", ".join(variaveis_validas) if variaveis_validas else "Nenhuma variável válida encontrada.")
#else:
#    st.warning("Nenhuma variável categórica com dados suficientes foi avaliada.")


