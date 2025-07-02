import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from io import StringIO

st.set_page_config(page_title="Análise ANOVA - Ames Housing", layout="wide")
st.title("Análise Estatística com ANOVA - Ames Housing Dataset")

#st.sidebar.header("Upload do Dataset")
uploaded_file = 'src/AmesHousing.csv'  # Default file for demonstration


df = pd.read_csv(uploaded_file)
st.success("Arquivo carregado com sucesso!")

 # Seção extra - Visualização das colunas
st.header("🔎 Visualização das Colunas do Dataset")
st.dataframe(pd.DataFrame({"Colunas": df.columns}))

# Seção extra - Visualização das colunas
st.header("🔎 Visualização das Colunas do Dataset")
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
colunas_df = pd.DataFrame({
    "Coluna": df.columns,
    "Descrição": [descricoes.get(col, "") for col in df.columns]
})
st.dataframe(colunas_df)



# Seleção das variáveis
var_target = 'SalePrice'
var1 = 'Neighborhood' # bairro
var2 = 'House_Style'  # estilo da casa
var3 = 'Bsmt_Full_Bath'   # banheiro completo no porão
#var3 = 'Heating'     # tipo de aquecimento

# Substitui espaços por underscore (_) em todas as colunas
df.columns = df.columns.str.replace(' ', '_')

        
df_clean = df[[var_target, var1, var2, var3]].dropna()

# Seção 1 - Análise Exploratória

st.header(f"1. Análise Exploratória das Três Características (Assumindo Normalidade): {var1}, {var2} e {var3}")
for var in [var1, var2, var3]:
    st.subheader(f"Distribuição de {var} vs Preço de Venda")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=var, y=var_target, data=df_clean, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# Seção 2 - ANOVA Onoway
st.markdown("## ANOVA: Análise de Variância")
st.header("2. ANOVA: Comparação de Preço Médio entre Níveis das Características")
results = {}
for var in [var1, var2, var3]:
    grupos = [group[var_target].values for name, group in df_clean.groupby(var)]
    anova = stats.f_oneway(*grupos)
    results[var] = anova.pvalue
    st.write(f"**{var}**: p-valor = {anova.pvalue:.4f}")

# Seção 3 - Interpretação dos Resultados
st.header("3. Interpretação dos Resultados da ANOVA")
for var in results:
    if results[var] < 0.05:
        st.success(f"Há diferença estatisticamente significativa nos preços médios entre os níveis de {var} (p-valor = {results[var]:.4f})")
    else:
        st.info(f"Não há evidência de diferença significativa entre os níveis de {var} (p-valor = {results[var]:.4f})")

# Seção 4 - Validação dos Pressupostos
st.header("4. Validação dos Pressupostos da ANOVA")
for var in [var1, var2, var3]:
    st.subheader(f"Pressupostos para {var}")
    modelo = sm.OLS.from_formula(f'{var_target} ~ C({var})', data=df_clean).fit()
    residuos = modelo.resid

    shapiro = stats.shapiro(residuos)
    bp_test = het_breuschpagan(residuos, modelo.model.exog)

    st.write(f"Shapiro-Wilk (normalidade dos resíduos): p-valor = {shapiro.pvalue:.4f}")
    st.write(f"Breusch-Pagan (homocedasticidade): p-valor = {bp_test[1]:.4f}")

# Seção 5 - Alternativas se ANOVA tradicional não for adequada
st.header("5. Alternativas à ANOVA Tradicional")
for var in [var1, var2, var3]:
    modelo = sm.OLS.from_formula(f'{var_target} ~ C({var})', data=df_clean).fit()
    residuos = modelo.resid
    shapiro = stats.shapiro(residuos)
    bp_test = het_breuschpagan(residuos, modelo.model.exog)

    if shapiro.pvalue < 0.05 or bp_test[1] < 0.05:
        grupos = [group[var_target].values for name, group in df_clean.groupby(var)]
        kruskal = stats.kruskal(*grupos)
        st.warning(f"Para {var}, ANOVA tradicional pode não ser adequada. Teste de Kruskal-Wallis: p-valor = {kruskal.pvalue:.4f}")
    else:
        st.success(f"Para {var}, os pressupostos foram atendidos. ANOVA tradicional é apropriada.")

# Seção 6 - Relatório Final
st.header("6. Relatório Final")
with st.expander("Clique para visualizar o relatório consolidado"):
    report = """
## Relatório Consolidado

### Variáveis analisadas:
- Bairro (Neighborhood)
- Tipo de Garagem (GarageType)
- Número de Lareiras (Fireplaces)

### Resultados da ANOVA:
- Neighborhood: {0:.4f}
- GarageType: {1:.4f}
- Fireplaces: {2:.4f}

### Interpretação:
- A ANOVA revelou diferenças estatisticamente significativas nos preços médios com base em algumas características, conforme os valores de p apresentados.

### Validação dos Pressupostos:
- Verificou-se normalidade e homocedasticidade para cada variável.
- Quando os pressupostos foram violados, optou-se pelo uso do teste de Kruskal-Wallis.

### Conclusão:
- As análises confirmam que certas características influenciam significativamente os preços das casas.
- A ANOVA foi utilizada quando adequada, e testes não-paramétricos foram aplicados como alternativa.
""".format(results[var1], results[var2], results[var3])
    st.text_area("Relatório Final", value=report, height=300)

# Seção 7 - Bibliografia
st.header("7. Bibliografia")
st.markdown("""
- Ames Housing Dataset: https://www.kaggle.com/datasets/prevek18/ames-housing-dataset  
- Montgomery, D.C. *Design and Analysis of Experiments*. Wiley.  
- Hair, J.F. et al. *Multivariate Data Analysis*. Pearson.  
- Scipy & Statsmodels Documentation
""")



st.warning("Por favor, envie o dataset CSV para começar a análise.")
