#!/usr/bin/env python
# coding: utf-8

# # Trabalho Prático Individual — Tópicos de Big Data em Python
# 
# **Aluno:** Theo Mencalha — **Matrícula:** 202502193254

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

CAMINHO_ORIGINAL = "dados/vendas_bigdata_2026.csv"
CAMINHO_TRATADO = "dados/vendas_tratadas.csv"


# ## 1. Inspeção inicial

# In[2]:


df_raw = pd.read_csv(CAMINHO_ORIGINAL)
df_raw.head()


# In[3]:


print("shape:", df_raw.shape)
df_raw.info()


# In[4]:


print(df_raw.dtypes)
print("\nValores ausentes por coluna:")
print(df_raw.isna().sum())
print("\nLinhas duplicadas (linha inteira):", df_raw.duplicated().sum())
print("pedido_id repetidos:", df_raw["pedido_id"].duplicated().sum())


# ### Respostas
# 
# **Quantos registros existem?** 1.200 registros e 15 colunas.
# 
# **Problemas de qualidade identificados**:
# - **8 linhas duplicadas**.
# - **Datas inválidas/ausentes:** 1 data nula e 4 textos que não são datas válidas (`2026-02-30`, `31/11/2026`, `2026-13-05`, `data_invalida`). A coluna `data` vem como texto.
# - **`regiao` com 12 nulos e palavres escritas de forma inconsistentes** (` Sudeste`, `sudeste`, `CENTRO-OESTE`, espaços extras etc.).
# - **`categoria` com 8 nulos**.
# - **`avaliacao` com 346 nulos** (~29%): clientes que não avaliaram.
# - **`quantidade` ≤ 0** (5 linhas) e **`preco_unitario` ≤ 0** (5 linhas).
# - **`valor` incoerente** com `quantidade × preco × (1 − desconto/100)` em 19 linhas.
# 
# **Primários ou secundários?** São dados secundários: foram coletados e registrados por terceiros (o sistema de vendas da empresa) para finalidade operacional e apenas *cedidos* a mim para análise; eu não os coletei nem controlei o processo de coleta.

# ## 2. Preparação e qualidade dos dados
# 
# Estratégia: trabalhar sobre uma cópia (`.copy()`), tratar cada problema em uma etapa, nunca apagar silenciosamente: registros inválidos vão para uma tabela de quarentena e as decisões são justificadas em cada etapa.

# ### 2.1 Conversão e validação da coluna de data

# In[5]:


df = df_raw.copy()
df["data_original"] = df["data"]
df["data"] = pd.to_datetime(df["data_original"], format="%Y-%m-%d", errors="coerce")

datas_invalidas = df.loc[df["data"].isna(), ["pedido_id", "data_original"]]
print("Datas inválidas ou ausentes:", len(datas_invalidas))
datas_invalidas


# **Decisão:** `errors="coerce"` transforma o que não é data em `NaT`, e o valor original fica guardado em `data_original` para auditoria.

# ### 2.2 Duplicatas

# In[6]:


colunas_originais = df_raw.columns.tolist()
dups = df.loc[df.duplicated(subset=colunas_originais, keep=False)].sort_values("pedido_id")
print("Linhas envolvidas em duplicação:", len(dups))
print("Duplicatas exatas a remover:", df.duplicated(subset=colunas_originais).sum())
print("pedido_id repetido com conteúdo diferente:",
      (df["pedido_id"].duplicated(keep=False) & ~df.duplicated(subset=colunas_originais, keep=False)).sum())
dups.head(4)


# In[7]:


df = df.drop_duplicates(subset=colunas_originais, keep="first").copy()
print("Registros após remover duplicatas:", len(df))


# **Decisão:** as 8 duplicatas são cópias idênticas (mesmo `pedido_id`, mesmos valores) e nenhum `pedido_id` se repete com conteúdo diferente. Um pedido não pode existir duas vezes, e manter inflaria o faturamento; removi mantendo a primeira ocorrência.

# ### 2.3 Padronização da coluna `regiao`

# In[8]:


print("Antes:", sorted(df["regiao"].dropna().unique()))
df["regiao"] = df["regiao"].str.strip().str.title()
print("Depois:", sorted(df["regiao"].dropna().unique()))


# **Decisão:** `strip()` remove espaços e `title()` unifica a caixa.

# ### 2.4 Valores ausentes

# In[9]:


print("Ausentes antes:\n", df[["data", "categoria", "regiao", "avaliacao"]].isna().sum())

# categoria: cada produto pertence a uma única categoria -> dá para recuperar com segurança
print("\nMax. de categorias por produto:", df.groupby("produto")["categoria"].nunique().max())
mapa_categoria = df.dropna(subset=["categoria"]).drop_duplicates("produto").set_index("produto")["categoria"]
mascara_cat = df["categoria"].isna()
df.loc[mascara_cat, "categoria"] = df.loc[mascara_cat, "produto"].map(mapa_categoria)

# regiao: não há como inferir -> rótulo explícito, sem descartar a venda
df.loc[df["regiao"].isna(), "regiao"] = "Não informado"

print("\nAusentes depois:\n", df[["data", "categoria", "regiao", "avaliacao"]].isna().sum())


# **Decisões:**
# - `categoria`: imputada pelo produto, pois a relação produto → categoria é 1:1 no próprio dataset (sem chute).
# - `regiao`: sem fonte para inferir; usei "Não informado" para não perder o faturamento e deixar a lacuna visível.
# - `data`: mantida como `NaT` (seção 2.1).
# - `avaliacao`: mantida nula. A ausência significa "cliente não avaliou" e a coluna não entra nas análises de faturamento; imputar média distorceria a nota real.

# ### 2.5 Quantidade ≤ 0, preço ≤ 0 e coerência do `valor`

# In[10]:


df["valor_esperado"] = (
    df["quantidade"]
    * df["preco_unitario"]
    * (1 - df["desconto_percentual"] / 100)
).round(2)
df["diferenca_valor"] = (df["valor"] - df["valor_esperado"]).round(2)

TOLERANCIA = 0.02  # arredondamento de centavos

df["qtd_invalida"] = df["quantidade"] <= 0
df["preco_invalido"] = df["preco_unitario"] <= 0
df["valor_incoerente"] = df["diferenca_valor"].abs() > TOLERANCIA
df["registro_invalido"] = df[["qtd_invalida", "preco_invalido", "valor_incoerente"]].any(axis=1)

print("quantidade <= 0      :", df["qtd_invalida"].sum())
print("preco_unitario <= 0  :", df["preco_invalido"].sum())
print("valor incoerente     :", df["valor_incoerente"].sum())
print("registros inválidos  :", df["registro_invalido"].sum(), "(união dos três critérios)")

colunas_diag = ["pedido_id", "quantidade", "preco_unitario", "desconto_percentual", "valor", "valor_esperado", "diferenca_valor"]
df.loc[df["registro_invalido"], colunas_diag].sort_values("diferenca_valor")


# **Decisão:** todos os registros com quantidade ≤ 0, preço ≤ 0 ou `valor` fora do esperado (tolerância de R$ 0,02 por arredondamento) são isolados em quarentena e excluídos das análises financeiras. Observações:
# - Os 10 casos de quantidade/preço inválidos também violam a regra de negócio.
# - Alguns casos parecem recuperáveis (ex.: preço negativo cujo `valor` bate com |preço|), mas não tem como saber qual campo está errado; corrigir seria adivinhar.

# In[11]:


df_quarentena = df.loc[df["registro_invalido"]].copy()
df_clean = df.loc[~df["registro_invalido"]].copy()

# Fluxo de registros
print("Originais              :", len(df_raw))
print("(-) duplicatas         :", len(df_raw) - len(df))
print("(-) quarentena         :", len(df_quarentena))
print("= base analítica       :", len(df_clean))
print("   dos quais sem data  :", df_clean["data"].isna().sum())


# In[12]:


# Salva a base tratada (insumo do PySpark). Colunas auxiliares de diagnóstico ficam de fora.
colunas_saida = colunas_originais + ["valor_esperado"]
df_clean[colunas_saida].to_csv(CAMINHO_TRATADO, index=False)
print("Base tratada salva em", CAMINHO_TRATADO, "->", df_clean.shape)


# ## 3. Análise com Pandas
# 
# **Definição:** *faturamento* = soma de `valor` (já líquido do desconto, sem frete), sobre a base tratada (`df_clean`).

# In[13]:


fat_regiao = df_clean.groupby("regiao")["valor"].sum().sort_values(ascending=False)
print("1) Faturamento por região:")
print(fat_regiao.round(2))
print("\n=> Maior faturamento:", fat_regiao.idxmax(), f"(R$ {fat_regiao.max():,.2f} | {fat_regiao.max()/fat_regiao.sum():.1%} do total)")


# In[14]:


top5_produtos = df_clean.groupby("produto")["valor"].sum().nlargest(5)
print("2) Top 5 produtos por faturamento:")
print(top5_produtos.round(2))


# In[15]:


fat_canal = df_clean.groupby("canal")["valor"].sum().sort_values(ascending=False)
print("3) Faturamento por canal:")
print(fat_canal.round(2))
print("\n=> Maior faturamento:", fat_canal.idxmax(), f"({fat_canal.max()/fat_canal.sum():.1%} do total)")


# In[16]:


pagamentos = df_clean["forma_pagamento"].value_counts()
print("4) Forma de pagamento (nº de pedidos):")
print(pagamentos)
print("\n=> Mais utilizada:", pagamentos.idxmax(), f"({pagamentos.max()/pagamentos.sum():.1%} dos pedidos)")


# In[17]:


# Evolução mensal (linhas sem data válida não entram)
df_mes = df_clean.dropna(subset=["data"]).copy()
df_mes["mes"] = df_mes["data"].dt.to_period("M")
mensal = df_mes.groupby("mes")["valor"].sum()
mensal_df = mensal.to_frame("faturamento")
mensal_df["var_mensal_%"] = (mensal_df["faturamento"].pct_change() * 100).round(1)
print("5) Faturamento mensal:")
print(mensal_df.round(2))
print("\nÚltima data na base:", df_mes["data"].max().date(), "-> novembro/2026 é um mês incompleto")


# ### Gráficos

# In[18]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: faturamento por região
(fat_regiao / 1000).sort_values().plot.barh(ax=axes[0], color="#2a6f97")
axes[0].set_title("Faturamento por região")
axes[0].set_xlabel("Faturamento (R$ mil)")
axes[0].set_ylabel("")
for i, v in enumerate((fat_regiao / 1000).sort_values()):
    axes[0].text(v + 2, i, f"{v:,.0f}", va="center", fontsize=9)

# Gráfico 2: evolução mensal (exclui nov/2026, mês incompleto)
mensal_completo = mensal[mensal.index < pd.Period("2026-11")]
axes[1].plot(mensal_completo.index.astype(str), mensal_completo.values / 1000, marker="o", color="#c1121f")
axes[1].set_title("Evolução mensal do faturamento (jan–out/2026)")
axes[1].set_ylabel("Faturamento (R$ mil)")
axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ### Conclusão
# 
# O faturamento bruto tratado soma cerca de R$ 535 mil em 1.173 pedidos válidos. O Sudeste lidera com R$ 221,7 mil (~41% do total), mais que o dobro do segundo colocado (Sul, R$ 111,6 mil). O Site é o canal dominante, com R$ 271 mil (~51%), à frente do App (R$ 188 mil) e da Loja física (R$ 76 mil). Cadeira e Mesa são os produtos que mais faturam (R$ 59,5 mil e R$ 56,8 mil), e o Pix é a forma de pagamento mais usada (~42% dos pedidos). Mês a mês não há tendência clara de crescimento: o faturamento oscila entre R$ 47 mil e R$ 55 mil até maio, cai em junho (R$ 40,5 mil) e atinge o pico em julho (R$ 67,5 mil); outubro (R$ 62 mil) também é forte. Cerca de 6% dos pedidos foram devolvidos (~R$ 34 mil), um custo a considerar. Recomenda-se investigar o que causou o pico de julho e a queda de junho, a baixa participação do Norte (R$ 37 mil), e corrigir na origem os problemas de qualidade (datas, região e valores).

# ## 4. PySpark DataFrame

# In[19]:


from pyspark.sql import SparkSession, functions as F

spark = (
    SparkSession.builder
    .appName("trabalho_bigdata")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")   # base pequena: poucas partições
    .config("spark.ui.enabled", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df_spark = spark.read.csv(CAMINHO_TRATADO, header=True, inferSchema=True)
df_spark.printSchema()
print("Registros no Spark:", df_spark.count())


# In[20]:


resumo_regioes = (
    df_spark
    .filter((F.col("quantidade") > 0) & (F.col("preco_unitario") > 0))   # transformação
    .groupBy("regiao")                                                   # transformação (wide)
    .agg(                                                                # transformação
        F.round(F.sum("valor"), 2).alias("faturamento_total"),
        F.countDistinct("pedido_id").alias("qtd_pedidos"),
        F.sum("quantidade").alias("qtd_itens"),
    )
    .orderBy(F.col("faturamento_total").desc())                          # transformação (wide)
)

resumo_regioes.show()   # AÇÃO: dispara a execução


# In[21]:


# Plano físico: observe o "Exchange hashpartitioning(regiao...)" = shuffle do groupBy
resumo_regioes.explain()


# In[22]:


# Exporta o resultado (toPandas também é uma ação) e valida contra o Pandas
resumo_pd = resumo_regioes.toPandas()
resumo_pd.to_csv("resultado/resumo_regioes.csv", index=False)

conferencia = df_clean.groupby("regiao")["valor"].sum().round(2)
assert (resumo_pd.set_index("regiao")["faturamento_total"].sort_index() - conferencia.sort_index()).abs().max() < 0.01
print("Spark e Pandas conferem. Arquivo salvo em resultado/resumo_regioes.csv")
resumo_pd


# ## 5. Spark SQL

# In[23]:


df_spark.createOrReplaceTempView("vendas")

# Pergunta: quais categorias têm a maior taxa de devolução nas vendas online (Site e App)?
consulta = """
SELECT
    categoria,
    COUNT(*)                                                   AS total_pedidos,
    SUM(CASE WHEN devolvido = 'Sim' THEN 1 ELSE 0 END)         AS pedidos_devolvidos,
    ROUND(100.0 * SUM(CASE WHEN devolvido = 'Sim' THEN 1 ELSE 0 END) / COUNT(*), 2) AS taxa_devolucao_pct
FROM vendas
WHERE canal IN ('Site', 'App')
GROUP BY categoria
ORDER BY taxa_devolucao_pct DESC
"""
spark.sql(consulta).show()


# Esta consulta responde uma pergunta diferente da seção 4.

# ## 6. HDFS e escalabilidade
# 
# Com um arquivo de vários TB no HDFS (fator de replicação 3), o arquivo é dividido em blocos (ex.: 128 MB), cada um replicado em 3 DataNodes.
# 
# - **NameNode:** é o "mestre" que guarda apenas os **metadados** (namespace, quais blocos formam cada arquivo e em quais DataNodes estão); não armazena dados e coordena leituras/escritas.
# - **DataNodes:** são os "trabalhadores" que **armazenam fisicamente os blocos** e os servem aos clientes, reportando ao NameNode a lista de blocos que possuem.
# - **Falta de heartbeat:** os DataNodes enviam heartbeats periódicos; se param de chegar por um tempo, o NameNode marca o nó como morto e replica os blocos que ele guardava a partir das cópias restantes, restaurando o fator 3.
# - **Por que a replicação ajuda:** cada bloco existe em 3 nós (em geral, em racks diferentes); se um nó ou rack falha, os dados continuam acessíveis em outras cópias (alta disponibilidade e tolerância a falhas) e leituras podem ser feitas na réplica mais próxima.

# ## 7. Auditoria de Machine Learning
# 
# ```python
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# ```
# 
# ### Problema: **vazamento de dados (data leakage)**
# 
# O `StandardScaler` é ajustado (`fit`) em todo o conjunto `X` *antes* da divisão treino/teste. A média e o desvio-padrão usados na padronização incluem, portanto, informação das linhas de teste, que deveriam ser "desconhecidas" durante o treino. O resultado é uma avaliação otimista: o desempenho medido no teste não representa o desempenho real com dados novos.
# 
# ### Correção
# Dividir primeiro e ajustar o scaler somente no treino; o teste apenas é transformado com os parâmetros do treino. Usar `Pipeline` garante isso automaticamente (inclusive dentro de validação cruzada, em que o scaler é reajustado em cada fold).

# In[24]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Dados sintéticos apenas para demonstrar que o código corrigido executa
X, y = make_classification(n_samples=500, n_features=10, random_state=42)

# 1) Divide ANTES de qualquer ajuste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Pipeline: o scaler é ajustado só com X_train; o teste só é transformado
modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])
modelo.fit(X_train, y_train)
print(f"Acurácia no teste: {modelo.score(X_test, y_test):.3f}")

# 3) Validação cruzada sem vazamento (scaler reajustado em cada fold)
scores = cross_val_score(modelo, X_train, y_train, cv=5)
print(f"CV (5 folds): {scores.mean():.3f} ± {scores.std():.3f}")


# In[25]:


spark.stop()

