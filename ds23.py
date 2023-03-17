import streamlit as st
import pandas as pd
import geopy.distance
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.pyplot as plt
import seaborn as sns

header = st.container()
dataset = st.container()
features = st.container()
model_training  = st.container()


@st.cache_data
def load_data(file_data):
    try:
        return pd.read_csv(file_data, dtype=dtypes,decimal=',',sep=';',parse_dates=['periodo'])
    except:
        return pd.read_csv(file_data, dtype=dtypes,decimal=',',sep=';',encoding='latin-1')#.encode('utf-8')
        


file_path_vendas = "vendas.csv" #"data/vendas.csv"
file_path_vendas_predita2 = "vendas_predita.csv"#"data/vendas_predita2.csv"
file_path_graf = "df_graf.csv"
with header:
    st.title ('Estudo de viabilidade econômica DS-23')
    st.header('Apresentação')
    st.write("A implantação de uma **dark store** tem se revelado como uma das estratégias mais eficientes adotadas por muitas empresas para atender às necessidades dos clientes em relação à entrega de produtos de forma rápida e segura.")
    st.write("Nesse contexto, a previsão de vendas através de técnicas de machine learning vem sendo utilizada em larga escala como ferramenta indispensável para a tomada de decisões em relação à abertura de novas filiais, especialmente para empresas de comércio eletrônico que buscam atender às demandas dos clientes em relação à entrega de produtos de forma rápida e segura;\n")
    st.write("O presente estudo busca suprir essa busca por informação através do desenvolvimento de modelos estatísticos elaborados a partir da análise de dados históricos de venda, projetando assim uma possível demanda comportamental dos consumidores axiliando a tomada de decisões mais precisas e seguras sobre expansão dos negócios da empresa.")

with dataset:
    st.header('Previsão de Vendas')
    st.write("Apesar de terem sido iniciadas em 2019, o volume de vendas através do comércio eletrônico ainda apresentava uma certa irregularidade, o que poderia vir a prejudicar a homogeneidade dos dados e consequentemente comprometer a análise. Sendo assim, optou-se por utilizar como base de dados as vendas mensais no período de **março/2021** a **fevereiro/2023**.") 
    st.write("Em seguida foram agregadas informações oficiais sobre as características dos municípios de Minas Gerais e Espírito Santo originadas de fontes oficiais do IBGE, CAGED e IPEA, totalizando 252 tipos diferentes de dados quantitativos e qualitativos sobre a população em diversos períodos.")
    st.write("Através o método *StepWise* foram selecionadas cerca de 16 variáveis para treinamento do modelo estatístico relacionadas a:")
    lista = ["Taxa de envelhecimento", "Educação", "Frequência escolar","% População rural", "IDH"]

    for item in lista:
        st.write("- " + item)
    
    
    dtypes = {"cod_IBGE": str, "cidade": str, "UF": str,"latitude": float,"longitude": float,"venda_predita": float }
    # df = pd.read_csv(file_path_vendas_predita2,dtype=dtypes,decimal=',',sep=';')
    df = load_data(file_path_vendas_predita2)
    dtypes = {"Cidade": str, "UF": str,"venda_total": float,"periodo": str  ,"venda_predita": float }
    usecols = ["Cidade", "UF","venda_total",'periodo']
    df_vendas = load_data(file_path_vendas)
    df_vendas['periodo'] = pd.to_datetime(df_vendas['periodo'])
    df_vendas['mes_ano'] = df_vendas['periodo'].dt.strftime('%Y-%m')

    df_vendas['Cidade']=df_vendas['Cidade'].str.upper()
    df_soma = df_vendas.groupby('mes_ano')['venda_total'].sum()

############################################
#### DEMONSTRAÇÃO DOS DADOS TRABALHADOS ####
############################################    

    st.markdown('**Dados utilizados**')
    st.write('Vendas mensais')
    # Grafico 1: vendas mensais
    st.bar_chart(df_soma,use_container_width=True,width=5)
    st.markdown('População')
    sel_col, disp_col = st.columns(2)
    with disp_col:
        qt_pop=st.slider("*População em mil habitantes*",min_value=10,max_value=300,value=100,step=10)
    df_graf = load_data(file_path_graf)
    plt.rc('figure', figsize=(12, 4))
    fig, axes = plt.subplots(1, 2)

    # Gráfico 2: venda por predominância da população
    ax1 = axes[0]
    sns.countplot(x="predominancia", data=df_graf, dodge=True, ax = ax1)
    ax1.yaxis.grid(False)
    ax1.set_ylabel("Qtde de vendas realizadas")
    ax1.set_xlabel("Predominância da população")

    # Gráfico 3: Venda total em relação à população
    df_graf3=df_graf.groupby(['Cidade','predominancia','populacao'])['venda_total'].mean().to_frame().reset_index()
    df_graf3=df_graf3[(df_graf3['Cidade']!='SÃO JOÃO DO JACUTINGA')&(df_graf3['Cidade']!='SANTO ANTÔNIO DO MANHUAÇU')]
    df_graf3=df_graf3[(df_graf3['populacao']<=qt_pop*1000)]
    ax2 = axes[1]
    ax2.scatter(df_graf3['populacao'], df_graf3['venda_total'])
    # ax2.set_yscale('log')
    ax2.set_xlabel('Venda média em relação à população')
    ax2.set_ylabel('Venda média')
    plt.subplots_adjust(wspace=.5, hspace=0.4)
    # plt.show()
    st.pyplot(fig)
    

with features:
    st.header('Metodologia aplicada')
    st.write('Para analisar o conjunto de dados, utilizou-se o algoritmo Random Forest, uma técnica de machine learning capaz de prever valores com base em um conjunto de variáveis explicativas. Esse algoritmo gera múltiplas árvores de regressão, obtendo a média das previsões de todas as árvores para produzir uma predição mais precisa e estável.') 
    st.write('Devido à sua capacidade de lidar com grandes conjuntos de dados e com um elevado número de variáveis, o Random Forest tem se destacado como uma das mais robustas e poderosas técnicas de aprendizado de máquina devido à:')
    lista = ["Alta precisão","Boa generalização","Robustez a dados ausentes e valores extremos","Seleção de recursos","Interpretabilidade ampla"]
    for item in lista:
             st.write("- " + item)
             
######################################
#### SIMULAÇÃO DE VENDAS PREDITAS ####
######################################

with model_training:
    st.header('Simulações')
    st.write('A seguir poderão ser feitas várias simulações de venda em cidades nos estados de **MG** e **ES**.')
    # st.write(df.sample(5))
    sel_col, disp_col = st.columns(2) 
    cidades = df.cidade.tolist()  
    
    with sel_col:
        cidade_selecionada = st.selectbox('Selecione uma cidade:', cidades)
        km_dist=sel_col.slider("Qual o raio em km você deseja obter para compor a região da cidade pesquisada?",min_value=10,max_value=200,value=50,step=10)
        
    filtro = (df['cidade'] == cidade_selecionada)
    lat_ref=np.asarray(df[filtro].latitude)[0]
    lon_ref=np.asarray(df[filtro].longitude)[0]
    distances = []
    for lat, lon in zip(df["latitude"], df["longitude"]):
        distance = geopy.distance.distance((lat_ref, lon_ref), (lat, lon)).km
        distances.append(distance)
    df["distancia"] = distances

    df_km = df[(df["distancia"] <= km_dist)&(df["distancia"] >= 0)&(df["cod_IBGE"]!= '313120')].copy()
    df_km['distancia'] = round(df_km['distancia'],0)
    # print(f'\nForam encontradas {df_km.cidade.value_counts().sum()} cidades num raio de {km_dist} km de {cidade_selecionada}')
    soma_vendas = df_km['venda_predita'].sum()
    
    df_pesq=df_km[["cidade", "distancia",'venda_predita']].sort_values('venda_predita',ascending=True)
    df_pesq['venda prevista'] = df_pesq['venda_predita'].apply(lambda x: '{:,.0f}'.format(x).replace(',', '.'))
    
    total = df_pesq['venda_predita'].sum()
    
    with disp_col:        
        st.write(df_pesq[['cidade','distancia','venda prevista']].set_index('cidade'))
    with sel_col:
        st.markdown('**Venda total projetada na região: R$ {:,.2f}**'.format(total).replace(',', '.'))
      
    st.markdown(f"**Mapa de vendas projetadas por região num raio de {km_dist} km**")
############################
#### HEATMAP DE VENDAS  ####
############################
  
    region = st.radio("Detalhe da visualização do mapa:",('Micro', 'Macro'),index=1)
    if region == 'Micro':
        mapa = folium.Map(location=[lat_ref, lon_ref], zoom_start=10)
    else:
        mapa = folium.Map(location=[-18.912998, -43.940933], zoom_start=6)

    data = df[['cidade', 'latitude', 'longitude', 'venda_predita', 'populacao', 'pib_percapita', 'predominancia_pop_rural', 'IDHM']]
    data['predominancia'] = data['predominancia_pop_rural'].apply(lambda x: 'Urbana' if x == 0 else 'Rural')
    # st.write(data.sample(5))

    ipa_loc = (data[(data['latitude'] == -19.7992) & (data['longitude'] == -41.7164)].index[0])
    data.drop(ipa_loc, inplace=True)
    colors = {'Urbana': 'blue', 'Rural': 'green'}

    # Criar lista de coordenadas e valores de vendas preditas
    coordenadas_vendas = data[['latitude', 'longitude', 'venda_predita']].values.tolist()

    # Adicionar camada de heatmap
    HeatMap(coordenadas_vendas, 
            max_val=max(data['venda_predita']),
            name='Venda projetada',
            control=True,
            show=True,
            min_opacity=0.30,
            radius=20).add_to(mapa)

    # Adicionar marcadores com clusterização para cada cidade
    mc = MarkerCluster(max_cluster_radius=km_dist,
                       spiderfy_on_max_zoom=False)

    for i in data.index:
        cidade = data['cidade'][i]
        lat = data['latitude'][i]
        lon = data['longitude'][i]
        vendas = data['venda_predita'][i]
        predominancia = data['predominancia'][i]
        popup_text = f"População: {data['populacao'][i]:,.0f}<br>PIB per capita: {data['pib_percapita'][i]:,.0f}<br>IDHM: {data['IDHM'][i]:,.2f}<br>Predominância: {data['predominancia'][i]}"
        popup = folium.Popup(popup_text, max_width=300)
        folium.Marker(location=[lat, lon], 
                      tooltip=f'{cidade}<br> R$ {vendas:,.2f}', 
                      popup=popup,
                      icon=folium.Icon(color=colors[predominancia])
                      ).add_to(mc)

    mc.add_to(mapa)

    folium_static(mapa)
