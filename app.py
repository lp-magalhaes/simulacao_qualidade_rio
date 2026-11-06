import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Gêmeo Digital Fluvial - Sistema de Controle Dinâmico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CARREGAMENTO RIGOROSO DOS MODELOS (.pkl) ---
@st.cache_resource
def carregar_modelos():
    modelos = {}
    arquivos_modelos = {
        'Turbidez': 'modelo_XGBoost_Turbidez.pkl',
        'Vazão': 'modelo_XGBoost_Vazão (m3_s).pkl',
        'Sólido Total': 'modelo_RandomForest_Sólido Total.pkl',
        'Ferro Total': 'modelo_RandomForest_Ferro Total.pkl',
        'Clorofila-a': 'modelo_SVR_Clorofila-a.pkl',
        'Carbono Orgânico Total': 'modelo_RandomForest_Carbono Orgânico Total.pkl'
    }
    
    arquivos_faltantes = []
    
    for parametro, arquivo in arquivos_modelos.items():
        if os.path.exists(arquivo):
            with open(arquivo, 'rb') as f:
                modelos[parametro] = pickle.load(f)
        else:
            arquivos_faltantes.append(arquivo)
            
    if arquivos_faltantes:
        st.error("🚨 **Erro Crítico de Inicialização:** Os seguintes arquivos de modelo (.pkl) obrigatórios não foram encontrados no repositório:")
        for arq in arquivos_faltantes:
            st.markdown(f"- `{arq}`")
        st.info("Por favor, faça o upload dos arquivos ausentes na raiz do seu repositório do GitHub para liberar o motor do Gêmeo Digital.")
        st.stop()
            
    return modelos

modelos = carregar_modelos()

# --- 2. BASE DE DADOS DE CHUVA HISTÓRICA (Consolidação de 24h na RAM) ---
@st.cache_data
def carregar_e_consolidar_chuva():
    nome_arquivo = 'dados_chuva.xlsx'
    if os.path.exists(nome_arquivo):
        df_horario = pd.read_excel(nome_arquivo)
        df_horario['data'] = pd.to_datetime(df_horario['data'])
        df_diario = df_horario.groupby(df_horario['data'].dt.date)['chuva_diaria'].sum().reset_index()
        df_diario['data'] = pd.to_datetime(df_diario['data'])
        return df_diario
    else:
        st.error(f"🚨 **Tabela de Entrada Ausente:** O arquivo `{nome_arquivo}` não foi localizado na raiz do repositório.")
        st.stop()

df_chuva = carregar_e_consolidar_chuva()

def calcular_acumulados_para_data(df, data_alvo):
    dt_alvo = pd.to_datetime(data_alvo)
    
    def obter_soma(dias):
        dt_inicio = dt_alvo - timedelta(days=dias-1)
        sub_df = df[(df['data'] >= dt_inicio) & (df['data'] <= dt_alvo)]
        return float(sub_df['chuva_diaria'].sum())
        
    return {
        'chuva_7_dias_acum': obter_soma(7),
        'chuva_15_dias_acum': obter_soma(15),
        'chuva_30_dias_acum': obter_soma(30),
        'chuva_45_dias_acum': obter_soma(45),
        'chuva_60_dias_acum': obter_soma(60)
    }

# --- 3. MOTOR DE INFERÊNCIA PURO (Apenas Modelos Reais) ---
def rodar_inferencia(parametro, inputs_calculados):
    modelo = modelos[parametro]
    
    if parametro in ['Turbidez', 'Vazão', 'Sólido Total', 'Ferro Total']:
        reacao_inputs = {
            'chuva_60_dias_acum': inputs_calculados['chuva_60_dias_acum'],
            'chuva_45_dias_acum': inputs_calculados['chuva_45_dias_acum'],
            'chuva_30_dias_acum': inputs_calculados['chuva_30_dias_acum']
        }
    else:
        reacao_inputs = {
            'chuva_30_dias_acum': inputs_calculados['chuva_30_dias_acum'],
            'chuva_15_dias_acum': inputs_calculados['chuva_15_dias_acum'],
            'chuva_7_dias_acum': inputs_calculados['chuva_7_dias_acum']
        }
                         
    df_input = pd.DataFrame([reacao_inputs])
    predicao = modelo.predict(df_input)
    return float(predicao[0])

# --- 4. INTERFACE GRÁFICA DO USUÁRIO ---
st.title("🌊 Gêmeo Digital de Corpos Hídricos")
st.markdown("### Motor de Simulação de Resiliência e Homeostase Ecossistêmica")

abas = st.tabs(["📅 Simulação Temporal (30 Dias)", "🕹️ Teste de Estresse por IA (Homeostase)"])

lista_parametros = ['Turbidez', 'Vazão', 'Sólido Total', 'Ferro Total', 'Clorofila-a', 'Carbono Orgânico Total']

# ABA 1: SIMULAÇÃO TEMPORAL
with abas[0]:
    st.header("Análise Dinâmica de Série Temporal")
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        data_minima = datetime(2025, 7, 11).date()
        data_maxima = (datetime(2026, 10, 5) - timedelta(days=30)).date()
        
        data_selecionada = st.date_input(
            "Selecione a Data de Referência:",
            value=datetime(2026, 4, 1).date(),
            min_value=data_minima,
            max_value=data_maxima
        )
        
        valores_acum_ref = calcular_acumulados_para_data(df_chuva, data_selecionada)
        st.metric("Chuva Acumulada (30d)", f"{valores_acum_ref['chuva_30_dias_acum']:.1f} mm")
        st.metric("Chuva Acumulada (60d)", f"{valores_acum_ref['chuva_60_dias_acum']:.1f} mm")
        
    with col_input2:
        datas_simuladas = [pd.to_datetime(data_selecionada) + timedelta(days=i) for i in range(31)]
        lista_resultados = []
        
        for dt in datas_simuladas:
            acumulados_dt = calcular_acumulados_para_data(df_chuva, dt)
            linha_chuva = df_chuva[df_chuva['data'] == dt]
            chuva_d = float(linha_chuva['chuva_diaria'].sum()) if not linha_chuva.empty else 0.0
            
            res_dt = {'Data': dt, 'Chuva_Diaria': chuva_d}
            for param in lista_parametros:
                res_dt[param] = rodar_inferencia(param, acumulados_dt)
            lista_resultados.append(res_dt)
            
        df_projecao = pd.DataFrame(lista_resultados)
        
        st.subheader(f"Status dos Parâmetros em {data_selecionada.strftime('%d/%m/%Y')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Vazão", f"{df_projecao.loc[0, 'Vazão']:.2f} m³/s")
        m2.metric("Turbidez", f"{df_projecao.loc[0, 'Turbidez']:.1f} NTU")
        m3.metric("Sólidos Totais", f"{df_projecao.loc[0, 'Sólido Total']:.1f} mg/L")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Ferro Total", f"{df_projecao.loc[0, 'Ferro Total']:.2f} mg/L")
        m5.metric("Clorofila-a", f"{df_projecao.loc[0, 'Clorofila-a']:.1f} µg/L")
        m6.metric("COT", f"{df_projecao.loc[0, 'Carbono Orgânico Total']:.1f} mg/L")

    st.markdown("---")
    st.subheader("📈 Comportamento Integrado e Dinâmico do Rio (Próximos 30 dias)")
    
    param_selecionados = st.multiselect(
        "Selecione os Parâmetros para Monitoramento do Fluxo:",
        options=lista_parametros,
        default=['Vazão', 'Turbidez', 'Carbono Orgânico Total']
    )
    
    if param_selecionados:
        fig_temporal = go.Figure()
        for idx, param in enumerate(param_selecionados):
            fig_temporal.add_trace(go.Scatter(
                x=df_projecao['Data'], y=df_projecao[param],
                mode='lines+markers', name=param,
                yaxis=f"y{idx+1}" if idx > 0 else "y"
            ))
            
        layout_axes = {}
        for idx in range(1, len(param_selecionados)):
            layout_axes[f"yaxis{idx+1}"] = dict(
                title=param_selecionados[idx], overlaying='y',
                side='right', position=1 - (idx * 0.05)
            )
            
        titulo_eixo_y = " / ".join(param_selecionados) if len(param_selecionados) <= 2 else "Valores dos Parâmetros"
        fig_temporal.update_layout(
            xaxis=dict(title="Cronograma da Simulação"),
            yaxis=dict(title=titulo_eixo_y),
            template="plotly_dark", **layout_axes
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

# ABA 2: TESTE DE CHOQUE REAL CONECTADO ÀS INTELIGÊNCIAS ARTIFICIAIS COM DATA ANCORA (01/01/2026)
with abas[1]:
    st.header("Simulador de Resiliência de IA - Resposta ao Impulso de Chuva")
    
    # Bloco explicativo sobre a ancoragem dos dados exigida
    st.info("""
    ℹ️ **Nota de Configuração do Sistema:** Este experimento utiliza a data real de **01/01/2026** como Linha de Base absoluta. 
    O motor calcula o histórico de chuvas acumuladas que antecederam essa data na planilha e executa os modelos de IA para definir as condições operacionais normais. 
    O estresse climático selecionado abaixo simula um pulso de perturbação ocorrendo estritamente nesta data de referência, permitindo avaliar em porcentagem (Δ%) o tempo exato de reação e amortecimento do ecossistema.
    """)
    
    col_ia1, col_ia2 = st.columns(2)
    with col_ia1:
        parametro_ia = st.selectbox("Parâmetro de Resiliência Analisado:", lista_parametros, key="param_ia")
    with col_ia2:
        volume_choque = st.slider("Volume Adicional de Chuva Injetado no Dia 5 (mm):", min_value=0.0, max_value=150.0, value=0.0, step=5.0)

    # 1. Definição da Data Âncora fixa e Extração do Histórico Real
    data_ancora = datetime(2026, 1, 1)
    dias_sim_ia = 70
    eixo_dias = np.arange(0, dias_sim_ia + 1)
    
    # Criamos séries temporais que herdam o passado real de dados que antecedeu 01/01/2026
    lista_df_controle = []
    lista_df_choque = []
    
    # Reconstrói a linha do tempo sintética futura preservando a janela de memória hidrológica
    for i in eixo_dias:
        dt_simulada = data_ancora + timedelta(days=int(i))
        
        # Cenário de Controle: Dias após 01/01/2026 seguem estáveis sem chuva adicional
        lista_df_controle.append({'data': dt_simulada, 'chuva_diaria': 0.0})
        
        # Cenário de Choque: Adiciona o pulso de estresse pontualmente no Dia 5 da simulação
        chuva_choque_dia = volume_choque if i == 5 else 0.0
        lista_df_choque.append({'data': dt_simulada, 'chuva_diaria': chuva_choque_dia})
        
    df_futuro_controle = pd.DataFrame(lista_df_controle)
    df_futuro_choque = pd.DataFrame(lista_df_choque)
    
    # Concatena o passado real do Excel com os cenários simulados para que as janelas móveis de 60 dias calculem corretamente
    df_completo_controle = pd.concat([df_chuva[df_chuva['data'] < data_ancora], df_futuro_controle]).reset_index(drop=True)
    df_completo_choque = pd.concat([df_chuva[df_chuva['data'] < data_ancora], df_futuro_choque]).reset_index(drop=True)

    # 2. Rodando as Predições e Calculando Desvios em Relação ao Controle Operacional
    variacoes_percentuais = []
    
    for t in eixo_dias:
        dt_atual = df_futuro_controle.loc[t, 'data']
        
        # Calcula acumulados buscando o passado real concatenado
        acum_controle = calcular_acumulados_para_data(df_completo_controle, dt_atual)
        acum_choque = calcular_acumulados_para_data(df_completo_choque, dt_atual)
        
        pred_controle = rodar_inferencia(parametro_ia, acum_controle)
        pred_choque = rodar_inferencia(parametro_ia, acum_choque)
        
        if pred_controle != 0:
            delta_p = ((pred_choque - pred_controle) / pred_controle) * 100
        else:
            delta_p = 0.0
        variacoes_percentuais.append(delta_p)
        
    variacoes_percentuais = np.array(variacoes_percentuais)

    # 3. Métricas de Tempo baseadas na IA
    dia_reacao = None
    for t in range(6, len(eixo_dias)):
        if abs(variacoes_percentuais[t]) > 0.1:
            dia_reacao = t - 5
            break
            
    dia_estabilizacao = 0
    for t in range(len(eixo_dias) - 1, 5, -1):
        if not (-5.0 <= variacoes_percentuais[t] <= 5.0):
            dia_estabilizacao = t - 5
            break

    c_ia1, c_ia2, c_ia3 = st.columns(3)
    c_ia1.metric("Impacto Máximo sobre o Equilíbrio", f"{variacoes_percentuais[np.argmax(np.abs(variacoes_percentuais))]:+.1f}%")
    c_ia2.metric("⏱️ Tempo de Reação do Modelo", f"{dia_reacao} dia" if dia_reacao is not None else "Imediato")
    c_ia3.metric("⏱️ Tempo de Retorno Estável (Homeostase)", f"{dia_estabilizacao} dias" if dia_estabilizacao > 0 else "Imediato")

    # 4. Gráfico do Sistema de Controle Baseado em IA (Eixo Y centrado em 0% como a referência estável)
    fig_ia = go.Figure()
    fig_ia.add_trace(go.Scatter(x=eixo_dias, y=np.zeros(len(eixo_dias)), mode='lines', name='Referência Normal (01/01/2026)', line=dict(color='white', dash='dash')))
    fig_ia.add_trace(go.Scatter(x=eixo_dias, y=variacoes_percentuais, mode='lines+markers', name=f'Desvio Dinâmico - {parametro_ia}', line=dict(color='#00ffcc', width=3)))
    
    fig_ia.add_trace(go.Scatter(
        x=np.concatenate([eixo_dias, eixo_dias[::-1]]),
        y=np.concatenate([np.full(len(eixo_dias), 5.0), np.full(len(eixo_dias), -5.0)[::-1]]),
        fill='toself', fillcolor='rgba(0, 255, 0, 0.05)', line=dict(color='rgba(0,255,0,0)'), name='Banda Operacional de Controle (±5%)'
    ))

    fig_ia.add_vline(x=5, line_width=1.5, line_dash="dot", line_color="red", annotation_text="Perturbação do Pulso")
    if dia_estabilizacao > 0:
        fig_ia.add_vline(x=5 + dia_estabilizacao, line_width=1.5, line_dash="dot", line_color="lime", annotation_text="Homeostase")

    fig_ia.update_layout(
        title=f"Curva de Controle de Resiliência de IA para {parametro_ia}",
        xaxis_title="Dias após o Cenário de Referência",
        yaxis_title="Desvio do Parâmetro (Δ %)",
        template="plotly_dark", height=520
    )
    st.plotly_chart(fig_ia, use_container_width=True)
