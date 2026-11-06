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

# --- 1. CARREGAMENTO DOS MODELOS (.pkl) ---
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
    
    for parametro, arquivo in arquivos_modelos.items():
        if os.path.exists(arquivo):
            with open(arquivo, 'rb') as f:
                modelos[parametro] = pickle.load(f)
        else:
            modelos[parametro] = "mock"
            
    return modelos

modelos = carregar_modelos()

# --- 2. CARREGAMENTO E CONSOLIDAÇÃO HORÁRIA DO EXCEL PARA A MEMÓRIA RAM ---
@st.cache_data
def carregar_e_consolidar_chuva():
    nome_arquivo = 'dados_chuva.xlsx'
    
    if os.path.exists(nome_arquivo):
        # Lê a tabela horária do repositório
        df_horario = pd.read_excel(nome_arquivo)
        
        # Mapeamento padrão esperado (Ajustável de acordo com as colunas reais do usuário)
        # Identifica dinamicamente ou assume 'data_hora' e 'chuva'
        col_tempo = [c for c in df_horario.columns if 'data' in c.lower() or 'time' in c.lower()][0]
        col_chuva = [c for c in df_horario.columns if 'chuva' in c.lower() or 'precip' in c.lower() or 'pms' in c.lower()][0]
        
        df_horario[col_tempo] = pd.to_datetime(df_horario[col_tempo])
        
        # Agrupamento e Soma das 24 horas de cada dia na memória
        df_diario = df_horario.groupby(df_horario[col_tempo].dt.date)[col_chuva].sum().reset_index()
        df_diario.columns = ['data', 'chuva_diaria']
        df_diario['data'] = pd.to_datetime(df_diario['data'])
    else:
        # Fallback de dados sintéticos caso o arquivo excel não esteja presente (Evita quebra no deploy inicial)
        data_inicio = datetime(2025, 7, 11)
        data_fim = datetime(2026, 10, 5)
        datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
        np.random.seed(42)
        chuva_diaria = np.random.exponential(scale=3.5, size=len(datas))
        chuva_diaria[np.random.choice(len(datas), 12)] = np.random.uniform(40, 90, 12)
        df_diario = pd.DataFrame({'data': datas, 'chuva_diaria': chuva_diaria})
        
    return df_diario

df_chuva = carregar_e_consolidar_chuva()

# --- FUNÇÕES AUXILIARES DE CÁLCULO DE ACUMULADOS ---
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

def rodar_inferencia(parametro, inputs_calculados):
    modelo = modelos.get(parametro)
    
    # 1. MAPEAMENTO ESTRUTURADO DE ACORDO COM O SEU ESCOPO ORIGINAL
    if parametro in ['Turbidez', 'Vazão', 'Sólido Total', 'Ferro Total']:
        reacao_inputs = {
            'chuva_60_dias_acum': inputs_calculados['chuva_60_dias_acum'],
            'chuva_45_dias_acum': inputs_calculados['chuva_45_dias_acum'],
            'chuva_30_dias_acum': inputs_calculados['chuva_30_dias_acum']
        }
    else:  # Clorofila-a e Carbono Orgânico Total
        reacao_inputs = {
            'chuva_30_dias_acum': inputs_calculados['chuva_30_dias_acum'],
            'chuva_15_dias_acum': inputs_calculados['chuva_15_dias_acum'],
            'chuva_7_dias_acum': inputs_calculados['chuva_7_dias_acum']
        }
                         
    if modelo != "mock" and modelo is not None:
        # 2. CONVERSÃO EXPLICITA PARA DATAFRAME (Resolve o TypeError)
        # Criamos uma linha de dados isolada encapsulando em uma lista
        df_input = pd.DataFrame([reacao_inputs])
        
        # Garante o retorno float puro para a matriz temporal
        predicao = modelo.predict(df_input)
        return float(predicao[0])
    else:
        # Lógica matemática substituta de contingência (Fallback)
        # Convertemos os valores em lista para manter retrocompatibilidade com o mock
        valores_lista = list(reacao_inputs.values())
        soma = sum(valores_lista)
        if parametro == 'Turbidez': return max(2.0, 5.0 + soma * 0.4)
        if parametro == 'Vazão': return max(0.5, 12.0 + soma * 0.15)
        if parametro == 'Sólido Total': return max(10.0, 35.0 + soma * 1.2)
        if parametro == 'Ferro Total': return max(0.01, 0.05 + soma * 0.003)
        if parametro == 'Clorofila-a': return max(0.1, 18.0 - (soma * 0.05))
        if parametro == 'Carbono Orgânico Total': return max(0.5, 3.0 + soma * 0.08)
    return 0.0

# --- ABA 1: SIMULAÇÃO TEMPORAL (30 DIAS) ---
with abas[0]:
    st.header("Análise Dinâmica de Série Temporal")
    col_input1, col_input2 = st.columns([1, 2])
    
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
        st.write("#### Entrada Acumulada Calculada")
        st.metric("Chuva Acumulada (7d)", f"{valores_acum_ref['chuva_7_dias_acum']:.1f} mm")
        st.metric("Chuva Acumulada (30d)", f"{valores_acum_ref['chuva_30_dias_acum']:.1f} mm")
        st.metric("Chuva Acumulada (60d)", f"{valores_acum_ref['chuva_60_dias_acum']:.1f} mm")
        
    with col_input2:
        datas_simuladas = [pd.to_datetime(data_selecionada) + timedelta(days=i) for i in range(31)]
        lista_resultados = []
        
        for dt in datas_simuladas:
            acumulados_dt = calcular_acumulados_para_data(df_chuva, dt)
            res_dt = {'Data': dt, 'Chuva_Diaria': float(df_chuva[df_chuva['data'] == dt]['chuva_diaria'].sum())}
            for param in modelos.keys():
                res_dt[param] = rodar_inferencia(param, acumulados_dt)
            lista_resultados.append(res_dt)
            
        df_projecao = pd.DataFrame(lista_resultados)
        
        st.subheader(f"Status dos Parâmetros em {data_selecionada.strftime('%d/%m/%Y')}")
        m1, m2, m3 = st.columns(3)
        idx_hoje = df_projecao[df_projecao['Data'] == pd.to_datetime(data_selecionada)].index[0]
        m1.metric("Vazão", f"{df_projecao.loc[idx_hoje, 'Vazão']:.2f} m³/s")
        m2.metric("Turbidez", f"{df_projecao.loc[idx_hoje, 'Turbidez']:.1f} NTU")
        m3.metric("Sólidos Totais", f"{df_projecao.loc[idx_hoje, 'Sólido Total']:.1f} mg/L")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Ferro Total", f"{df_projecao.loc[idx_hoje, 'Ferro Total']:.3f} mg/L")
        m5.metric("Clorofila-a", f"{df_projecao.loc[idx_hoje, 'Clorofila-a']:.1f} µg/L")
        m6.metric("COT", f"{df_projecao.loc[idx_hoje, 'Carbono Orgânico Total']:.1f} mg/L")

    st.markdown("---")
    st.subheader("📈 Comportamento Integrado e Dinâmico do Rio (Próximos 30 dias)")
    
    param_selecionados = st.multiselect(
        "Selecione os Parâmetros para Monitoramento Integrado:",
        options=list(modelos.keys()),
        default=['Vazão', 'Turbidez', 'Carbono Orgânico Total']
    )
    
    if param_selecionados:
        fig_temporal = go.Figure()
        for idx, param in enumerate(param_selecionados):
            fig_temporal.add_trace(go.Scatter(
                x=df_projecao['Data'], 
                y=df_projecao[param],
                mode='lines+markers',
                name=param,
                yaxis=f"y{idx+1}" if idx > 0 else "y"
            ))
            
        layout_axes = {}
        for idx in range(1, len(param_selecionados)):
            layout_axes[f"yaxis{idx+1}"] = dict(
                title=param_selecionados[idx],
                overlaying='y',
                side='right',
                position=1 - (idx * 0.05),
                showgrid=False
            )
            
        fig_temporal.update_layout(
            xaxis=dict(title="Cronograma da Simulação (Projeção Temporal)"),
            yaxis=dict(title=param_selecionados[0]),
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40),
            **layout_axes
        )
        st.plotly_chart(fig_temporal, use_container_width=True)

# --- ABA 2: CÁLCULO DE HOMEOSTASE E TEMPO DE RECUPERAÇÃO ---
with abas[1]:
    st.header("Simulador de Choque e Resiliência de Sistemas de Controle")
    st.markdown("Modela a estabilidade homeostática do ecossistema simulando uma perturbação de pulso controlado.")
    
    col_par1, col_par2, col_par3 = st.columns(3)
    with col_par1:
        parametro_estudo = st.selectbox("Parâmetro Biogequímico de Estudo:", list(modelos.keys()))
    with col_par2:
        valor_estabilidade = st.number_input("Definir Linha de Base (Condição Normal do Rio):", value=20.0, step=1.0)
    with col_par3:
        magnitude_choque = st.slider("Magnitude do Estresse Climático (Pico do Pulso):", 1.5, 5.0, 2.5, step=0.1)
        
    dias_estudo = 60
    eixo_tempo = np.arange(0, dias_estudo + 1)
    linha_de_base = np.full(len(eixo_tempo), valor_estabilidade)
    curva_resposta = []
    
    dia_pulso_inicio = 5
    dia_pulso_fim = 8
    
    for t in eixo_tempo:
        if t < dia_pulso_inicio:
            curva_resposta.append(valor_estabilidade)
        elif dia_pulso_inicio <= t <= dia_pulso_fim:
            fator_subida = (t - dia_pulso_inicio) / (dia_pulso_fim - dia_pulso_inicio)
            valor_atual = valor_estabilidade + (valor_estabilidade * (magnitude_choque - 1) * Fator_subida)
            curva_resposta.append(valor_atual)
        else:
            t_decorrido = t - dia_pulso_fim
            decaimento = np.exp(-0.15 * t_decorrido) * np.cos(0.2 * t_decorrido)
            valor_atual = valor_estabilidade + (valor_estabilidade * (magnitude_choque - 1) * decaimento)
            curva_resposta.append(valor_atual)
            
    curva_resposta = np.array(curva_resposta)
    
    limiar_superior = valor_estabilidade * 1.05
    limiar_inferior = valor_estabilidade * 0.95
    
    tempo_recuperacao = None
    for t in range(len(eixo_tempo) - 1, dia_pulso_fim, -1):
        if not (limiar_inferior <= curva_resposta[t] <= limiar_superior):
            tempo_recuperacao = (t + 1) - dia_pulso_fim
            break
            
    if tempo_recuperacao is None or tempo_recuperacao <= 0:
        tempo_recuperacao = 0
        
    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    c_kpi1.metric("Pico Registrado no Distúrbio", f"{curva_resposta.max():.2f}")
    c_kpi2.metric("Faixa de Controle Estável (±5%)", f"{limiar_inferior:.2f} a {limiar_superior:.2f}")
    
    if tempo_recuperacao > 0:
        c_kpi3.metric("⏱️ Tempo de Recuperação", f"{tempo_recuperacao} dias")
    else:
        c_kpi3.metric("⏱️ Tempo de Recuperação", "Imediato / Estável")
        
    fig_homeostase = go.Figure()
    fig_homeostase.add_trace(go.Scatter(x=eixo_tempo, y=curva_resposta, mode='lines+markers', name='Resposta Dinâmica do Rio', line=dict(color='#00ffcc', width=3)))
    fig_homeostase.add_trace(go.Scatter(x=eixo_tempo, y=linha_de_base, mode='lines', name='Linha de Base (Equilíbrio)', line=dict(color='white', dash='dash')))
    
    fig_homeostase.add_trace(go.Scatter(
        x=np.concatenate([eixo_tempo, eixo_tempo[::-1]]),
        y=np.concatenate([np.full(len(eixo_tempo), limiar_superior), np.full(len(eixo_tempo), limiar_inferior)[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.07)',
        line=dict(color='rgba(0,255,0,0)'),
        hoverinfo='skip',
        name='Banda de Homeostase (±5%)'
    ))
    
    fig_homeostase.add_vline(x=dia_pulso_fim, line_width=1.5, line_dash="dot", line_color="orange", annotation_text="Fim do Evento")
    if tempo_recuperacao > 0:
        fig_homeostase.add_vline(x=dia_pulso_fim + tempo_recuperacao, line_width=1.5, line_dash="dot", line_color="lime", annotation_text="Estabilização")
        
    fig_homeostase.update_layout(
        title=f"Curva de Resiliência do Sistema de Controle - {parametro_estudo}",
        xaxis_title="Dias de Simulação",
        yaxis_title=parametro_estudo,
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig_homeostase, use_container_width=True)
