import json
import logging
import os
from textwrap import dedent
from typing import Any, Dict, List

import streamlit as st
import pandas as pd

# Tentativi di import per i due motori
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

logger = logging.getLogger("BurryInvestingPro")

# ==========================================================================
# PROMPT DI SISTEMA (identico)
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
# MODELLO LOCALE (SmolLM2-135M)
# ==========================================================================
@st.cache_resource(show_spinner=False)
def load_local_model():
    """Scarica e carica il modello SmolLM2-135M in formato GGUF."""
    if not LLAMA_AVAILABLE:
        return None
    try:
        model_filename = "smollm2-135m-instruct-q8_0.gguf"
        repo_id = "HackNetAyush/smollm2-135M-instruct-gguf-q8"
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        return Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=2,
            n_batch=512,
            verbose=False
        )
    except Exception as e:
        logger.error(f"Errore caricamento modello locale: {e}")
        return None

def ask_local_fallback(context: Dict[str, Any], user_question: str, mode: str = "Entrambi") -> str:
    """Usa SmolLM2 locale come fallback quando Gemini fallisce."""
    if not LLAMA_AVAILABLE:
        return "⚠️ **Backup locale non disponibile**: librerie mancanti."
    
    with st.spinner("Caricamento AI locale (primo avvio, pochi secondi)..."):
        llm = load_local_model()
    
    if llm is None:
        return "⚠️ **Impossibile caricare il modello locale**. Riprova."
    
    system_prompt, user_prompt = build_ai_messages(context, user_question, mode)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=512,
            stop=["</s>", "User:", "Assistant:"]
        )
        answer = response['choices'][0]['message']['content'].strip()
        if not answer:
            answer = "Il modello non ha generato una risposta valida."
        return answer + "\n\n---\n*🧠 Risposta generata da AI locale (SmolLM2-135M).*"
    except Exception as e:
        logger.error(f"Errore inferenza locale: {e}")
        return f"⚠️ **Errore nell'AI locale**: {e}"

# ==========================================================================
# FUNZIONE PRINCIPALE A CASCATA
# ==========================================================================
def ask_gemini_ticker_chat(context: Dict[str, Any], user_question: str, mode: str = "Entrambi", max_tokens: int = 8192) -> str:
    """
    Tenta prima con Google Gemini. Se fallisce (qualsiasi errore), usa SmolLM2 locale.
    Se Gemini non è configurato, usa direttamente il modello locale.
    """
    # --------------------------------------------------------------
    # 1. TENTATIVO CON GEMINI (se disponibile e configurato)
    # --------------------------------------------------------------
    api_key = os.getenv("GEMINI_API_KEY") or safe_get_secret("GEMINI_API_KEY", None)
    if GEMINI_AVAILABLE and api_key:
        try:
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
            logger.warning(f"Gemini fallito: {error_msg}")
            if "429" in error_msg or "quota" in error_msg.lower():
                st.info("⚠️ Quota Gemini esaurita. Attivazione AI locale...")
            else:
                st.info(f"⚠️ Errore Gemini ({error_msg[:100]}). Attivazione AI locale...")
    
    # --------------------------------------------------------------
    # 2. FALLBACK: MODELLO LOCALE (SmolLM2)
    # --------------------------------------------------------------
    return ask_local_fallback(context, user_question, mode)

# ==========================================================================
# CONTESTO PER LA SIDEBAR (VqAi)
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