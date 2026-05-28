import json
import logging
import os
import random
import time
from textwrap import dedent
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

# Tentativo di import per il backup locale con transformers
try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger("BurryInvestingPro")

# ==========================================================================
# PROMPT DI SISTEMA (identico a prima)
# ==========================================================================
SYSTEM_PROMPT = dedent("""
Sei l'assistente AI interno di V-Quant Pro, un'app Python/Streamlit di analisi finanziaria e portafoglio.

RUOLO
Agisci come analista finanziario rigoroso, prudente e argomentativo.
Non sei un indovino, non fai previsioni certe, non prometti rendimenti, non inventi dati.
Il tuo compito è interpretare in modo chiaro e approfondito i dati già calcolati dall'applicazione.

CONTESTO OPERATIVO
Riceverai in input un contesto strutturato generato dall'app, che può contenere:
- dati fondamentali, tecnici, metriche quantitative, rischio, timing, smart quant score.
- esito dei modelli classico/evoluto/personalizzabile, simulazioni Monte Carlo.
- dati di portafoglio, allocazione, concentrazione, correlazioni, beta, alpha, fiscalità, FX.

ANALISI INTEGRATA (DATI + CONOSCENZA ESTERNA)
- Usa i dati presenti nel contesto come base prioritaria per l'analisi quantitativa.
- Se i dati nel contesto sono parziali o per arricchire l'analisi professionale, sei autorizzato e incoraggiato a utilizzare le tue conoscenze aggiornate su mercati, macroeconomia e news recenti.
- Non limitarti a dire 'dato non disponibile'; se un parametro manca, commenta il titolo in base al settore di appartenenza e ai trend di mercato attuali.
- Non fornire consulenza finanziaria personalizzata definitiva. Non usare un linguaggio da guru.

STILE DELLE RISPOSTE
Le risposte devono essere: approfondite, precise, ben argomentate, ordinate in sezioni, scritte in italiano professionale.
LUNGHEZZA MINIMA: produci almeno 8-12 paragrafi brevi oppure sezioni equivalenti. Spiega sempre il perché delle conclusioni e collega ogni giudizio a metriche specifiche.

PRINCIPIO DI ANALISI
Ogni conclusione deve derivare da evidenze numeriche. Per ogni giudizio:
1. cita la metrica rilevante; 2. spiega cosa misura; 3. interpreta il valore osservato; 4. collega alle implicazioni pratiche; 5. segnala eventuali limiti.

QUANDO ANALIZZI UN SINGOLO TITOLO (Valuta se disponibili):
1) Qualità del business e fondamentali (ROIC, PEG, Debt/Equity, Revenue Growth, Net Margin, FCF Margin, Interest Coverage, Altman Z-Score).
2) Profilo quantitativo e rischio-rendimento (Sharpe, Sortino, Calmar, CAGR, Max Drawdown, Volatilità, VaR/CVaR, Skew, Kurtosis, R-Squared, Omega Ratio, Ulcer Index).
3) Timing e analisi tecnica (Timing Score, SMA50/200, RSI, MACD, Bollinger).
4) Esito sintetico: integra fondamentali, timing e quant. Segnala i segnali contrastanti.
5) Limiti dell'analisi: cosa suggeriscono i dati e cosa NON possono garantire.

QUANDO ANALIZZI UN PORTAFOGLIO (Valuta se disponibili):
1) Struttura (pesi, asset class, geografia, valuta).
2) Metriche aggregate (Rendimento, Volatilità, Sharpe, Beta, Alpha, Correlazione).
3) Concentrazione (HHI, ENS, Top1/Top3).
4) Coerenza del rischio (Drawdown vs CAGR).
5) Ribilanciamento e allocazione.
6) Effetti operativi (FX, PMC, Fiscalità).

GESTIONE DELLE CONTRADDIZIONI
Le contraddizioni sono centrali: evidenzia se, ad esempio, i fondamentali sono forti ma lo Sharpe è debole o se il timing è buono ma la valutazione non è attraente.

FORMATO STANDARD DELLA RISPOSTA
1. Sintesi iniziale
2. Lettura dei fondamentali
3. Lettura quantitativa e rischio
4. Lettura tecnica e timing
5. Integrazione dei segnali
6. Rischi principali
7. Limiti dell’analisi
8. Conclusione operativa prudente
""").strip()

USER_PROMPT_TEMPLATE = dedent("""
Analizza il seguente contesto prodotto dall'app.

ISTRUZIONI ADDIZIONALI:
- Non essere sintetico. Voglio una risposta approfondita e ben motivata.
- Usa tutte le metriche disponibili nel contesto. Quando una metrica manca, dillo.
- Se trovi conflitti tra segnali fondamentali, tecnici e quantitativi, evidenziali.
- Non fare previsioni certe, ma interpreta probabilità, qualità del profilo rischio/rendimento e robustezza del setup.

Vincolo di qualità: non chiudere la risposta finché non hai commentato esplicitamente:
- redditività, leva finanziaria, qualità dei margini, rendimento corretto per il rischio, drawdown, rischio di coda, struttura del trend, coerenza tra i segnali e principali fattori di fragilità.

DOMANDA UTENTE:
{user_question}

CONTESTO APP:
{json_context}
""").strip()

# ==========================================================================
# FUNZIONI DI SUPPORTO
# ==========================================================================
def safe_get_secret(key: str, default=None):
    env_val = os.getenv(key)
    if env_val:
        return env_val.strip()
    try:
        if key in st.secrets:
            val = st.secrets[key]
            return str(val).strip() if val is not None else default
    except Exception:
        pass
    return default

def build_ai_messages(context: Dict[str, Any], user_question: str, mode: str = "Entrambi") -> tuple[str, str]:
    json_context = json.dumps(context, ensure_ascii=False, default=str, indent=2)
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT_TEMPLATE.format(
        user_question=user_question,
        json_context=json_context,
    )
    if mode:
        user_prompt = f"Modalità modello: {mode}\n\n" + user_prompt
    return system_prompt, user_prompt

def build_ai_context_for_ticker(ticker: str, row: pd.Series, qm: Dict[str, Any], risk: Dict[str, Any],
                                score: float, reasons: List[str], mode: str) -> Dict[str, Any]:
    row_dict = row.to_dict() if row is not None and hasattr(row, 'to_dict') else dict(row or {})
    return {
        'ticker': ticker,
        'mode': mode,
        'timing_score': float(score or 0.0),
        'timing_reasons': reasons or [],
        'fundamentals': {
            'company_name': row_dict.get('Company Name'),
            'price': row_dict.get('Price'),
            'roic': row_dict.get('ROIC'),
            'peg_ratio': row_dict.get('PEG Ratio'),
            'debt_to_equity': row_dict.get('Debt/Equity'),
            'fcf_margin': row_dict.get('FCF Margin'),
            'net_margin': row_dict.get('Net Margin'),
            'revenue_growth': row_dict.get('Revenue Growth'),
        },
        'quant': qm or {},
        'risk': risk or {},
    }

# ==========================================================================
# FUNZIONE PRINCIPALE CON BACKUP LOCALE (transformers)
# ==========================================================================
def ask_gemini_ticker_chat(context: Dict[str, Any], user_question: str, mode: str = "Entrambi", max_tokens: int = 8192) -> str:
    """
    Tenta prima con Google Gemini (se API key configurata).
    In caso di errore 429 (quota esaurita) o fallimento, utilizza un modello locale
    (distilgpt2) tramite transformers come backup gratuito e senza limiti.
    """
    # --------------------------------------------------------------
    # 1. TENTATIVO CON GEMINI API
    # --------------------------------------------------------------
    api_key = os.getenv("GEMINI_API_KEY") or safe_get_secret("GEMINI_API_KEY", None)
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            system_prompt, user_prompt = build_ai_messages(context, user_question, mode)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=min(max_tokens, 8192)
                )
            )
            if response.text and response.text.strip():
                return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                logger.warning(f"Quota Gemini esaurita, passo al backup locale. Errore: {error_msg}")
                st.info("⚠️ Limite di Gemini raggiunto. Attivazione AI locale (gratuita)...")
            else:
                logger.warning(f"Errore generico Gemini: {error_msg}, passo al backup.")
                st.info("⚠️ Errore con Gemini. Attivazione AI locale...")

    # --------------------------------------------------------------
    # 2. FALLBACK: MODELLO LOCALE CON TRANSFORMERS
    # --------------------------------------------------------------
    if not TRANSFORMERS_AVAILABLE:
        return (
            "⚠️ **Backup locale non disponibile**\n\n"
            "Le librerie necessarie (transformers, torch) non sono installate.\n"
            "Per usare il backup, aggiungi `transformers>=4.46.0` e `torch>=2.5.0` al requirements.txt."
        )

    try:
        # Inizializza il generatore di testo locale (una sola volta, in cache)
        if "local_llm" not in st.session_state:
            with st.spinner("Caricamento AI locale (prima volta, richiede ~10-15 secondi)..."):
                # Usa un modello molto leggero: distilgpt2
                st.session_state.local_llm = pipeline(
                    "text-generation",
                    model="distilgpt2",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=1.15
                )
        generator = st.session_state.local_llm
        system_prompt, user_prompt = build_ai_messages(context, user_question, mode)
        # Formatta il prompt per il modello locale
        full_prompt = f"Sistema: {system_prompt}\nUtente: {user_prompt}\nAssistente:"
        # Genera la risposta
        result = generator(full_prompt, max_new_tokens=300)[0]['generated_text']
        # Estrae solo la parte dopo "Assistente:"
        if "Assistente:" in result:
            ai_reply = result.split("Assistente:", 1)[-1].strip()
        else:
            ai_reply = result.replace(full_prompt, "").strip()
        if not ai_reply:
            ai_reply = "L'AI locale non ha generato una risposta valida. Riprova."
        # Aggiunge una nota informativa
        return ai_reply + "\n\n---\n*🧠 Risposta generata da AI locale (gratuita, senza limiti).*"
    except Exception as e:
        logger.error(f"Errore con il modello AI locale: {e}")
        return (
            "⚠️ **Errore nel backup locale**\n\n"
            f"Dettaglio tecnico: {e}\n\n"
            "Riprova più tardi o configura una chiave API Gemini funzionante."
        )

# ==========================================================================
# FUNZIONE PER IL CONTESTO DELLA SIDEBAR (VqAi)
# ==========================================================================
def build_burry_ai_context(symbol: str, asset_type: str, mode: str = "Entrambi") -> Dict[str, Any]:
    symbol_clean = (symbol or "").upper().strip()
    context = {
        "ticker": symbol_clean,
        "asset_type": asset_type,
        "mode": mode,
        "note": "Usa la logica del programma per rispondere su azioni o ETF; se i dati completi non sono disponibili, dichiaralo esplicitamente."
    }
    try:
        live_map = st.session_state.get('burry_ai_live_context', {}) or {}
        if symbol_clean and symbol_clean in live_map:
            context['live_program_analysis'] = live_map[symbol_clean]
    except Exception as e:
        logger.debug(f'VqAi live context fallback: {e}')
    try:
        df = st.session_state.get('batch_results')
        if df is not None and not df.empty and 'Ticker' in df.columns and symbol_clean:
            row_match = df[df['Ticker'].astype(str).str.upper() == symbol_clean]
            if not row_match.empty:
                row = row_match.iloc[0]
                row_dict = row.to_dict()
                context['program_data'] = {
                    'company_name': row_dict.get('Company Name'),
                    'price': row_dict.get('Price'),
                    'pe_ratio': row_dict.get('P/E Ratio'),
                    'peg_ratio': row_dict.get('PEG Ratio'),
                    'roic': row_dict.get('ROIC'),
                    'debt_to_equity': row_dict.get('Debt/Equity'),
                    'fcf_margin': row_dict.get('FCF Margin'),
                    'net_margin': row_dict.get('Net Margin'),
                    'revenue_growth': row_dict.get('Revenue Growth'),
                    'altman_zscore': row_dict.get('Altman Z-Score'),
                    'piotroski_fscore': row_dict.get('F-Score'),
                }
    except Exception as e:
        logger.debug(f'VqAi context fallback su batch_results: {e}')
    try:
        selected = (st.session_state.get('selected_ticker') or '').upper().strip()
        context['selected_ticker_match'] = bool(selected and selected == symbol_clean)
    except Exception:
        context['selected_ticker_match'] = False
    return context