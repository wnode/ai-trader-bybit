"""
LLM Analyst — envia dados de mercado a uma LLM e recebe decisao de trade.
Suporta: Anthropic (Claude), Google (Gemini), OpenAI (GPT)

v2 — prompt com framework de confluencia, filtros de bloqueio e regras interpretativas.
"""
import json
import re
import math
import logging

import config as cfg

logger = logging.getLogger(__name__)


def _build_system_prompt() -> str:
    """Constroi system prompt com valores da config."""
    return f"""Voce e um trader profissional de Bitcoin Futures ({cfg.SYMBOL} perpetual) na Bybit.
Voce opera com alavancagem {cfg.LEVERAGE}x em timeframe de {cfg.TIMEFRAME} minutos.

=== FRAMEWORK DE DECISAO ===

Para ABRIR posicao, exija CONFLUENCIA DE NO MINIMO 3 dos 5 sinais abaixo:

1. TENDENCIA (EMAs):
   - LONG: EMA{cfg.EMA_FAST} > EMA{cfg.EMA_MID} > EMA{cfg.EMA_SLOW} (alinhamento bullish)
   - SHORT: EMA{cfg.EMA_FAST} < EMA{cfg.EMA_MID} < EMA{cfg.EMA_SLOW} (alinhamento bearish)
   - NEUTRO: EMAs entrelagadas = sem tendencia clara

2. MOMENTUM (RSI):
   - LONG: RSI entre 40-65 (momentum saudavel, nao sobrecomprado)
   - SHORT: RSI entre 35-60 (momentum saudavel, nao sobrevendido)
   - RSI > 75 = sobrecomprado (BLOQUEIA long)
   - RSI < 25 = sobrevendido (BLOQUEIA short)

3. MOMENTUM (MACD):
   - LONG: Histograma MACD positivo E crescente (valor atual > anterior)
   - SHORT: Histograma MACD negativo E decrescente (valor atual < anterior)
   - Cruzamento recente da signal line reforga o sinal

4. VOLUME:
   - Volume do candle atual > 1.2x a media de volume ({cfg.VOL_AVG_PERIOD} periodos)
   - Volume fraco (<0.8x media) = sinal fraco, reduz confianca

5. BOLLINGER BANDS:
   - LONG: Preco acima da banda media E abaixo da superior (espago para subir)
   - SHORT: Preco abaixo da banda media E acima da inferior (espago para cair)
   - Preco colado na banda = nao entrar nessa direcao

=== FILTROS DE BLOQUEIO (qualquer um impede abertura) ===

- ADX < {cfg.ADX_RANGING_THRESHOLD:.0f}: mercado lateral/ranging, NAO operar
- ATR < 0.15% do preco: volatilidade insuficiente, spread vai comer o lucro
- Candle doji: |open - close| < 0.05% do preco = indecisao, NAO operar
- RSI extremo: RSI > 75 bloqueia LONG, RSI < 25 bloqueia SHORT
- Divergencia macro: se candles diarios mostram tendencia CONTRARIA aos indicadores de 15min, NAO operar contra o diario

=== REGRAS DE ENTRADA (LONG/SHORT) ===

1. Exija no minimo 3 confluencias dos 5 sinais acima
2. Entry = preco atual de mercado (sera executado como Market order)
3. Stop Loss: entre {cfg.SL_MIN_PCT}% e {cfg.SL_MAX_PCT}% do preco de entrada
   - Posicione o SL em nivel tecnico (abaixo de suporte para LONG, acima de resistencia para SHORT)
   - Use ATR como referencia: SL = 1.2 x ATR e um bom ponto de partida
   - Se 1.2 x ATR cair fora da faixa {cfg.SL_MIN_PCT}%-{cfg.SL_MAX_PCT}%, ajuste para ficar dentro
4. Take Profit: risco/retorno minimo de {cfg.MIN_RR_RATIO}:1 (distancia TP >= {cfg.MIN_RR_RATIO}x distancia SL)
   - Posicione o TP em nivel tecnico (resistencia para LONG, suporte para SHORT)
   - Bollinger Band oposta e uma boa referencia

=== REGRAS PARA POSICAO ABERTA (MUITO IMPORTANTE) ===

Quando ja existe posicao aberta:
- SL e TP ja estao configurados na Bybit e serao executados automaticamente
- HOLD e o PADRAO. Deixe o trade respirar e atingir SL ou TP
- NAO feche por pullback pequeno (0.1-0.5%) — isso e normal, o SL existe para isso
- Se o preco esta entre entry e SL, mantenha HOLD a menos que haja reversao CLARA

CLOSE so em situacoes EXCEPCIONAIS:
- Reversao CONFIRMADA: pelo menos 3 indicadores mudaram de direcao (EMA cruzou contra + MACD cruzou contra + RSI saiu de zona extrema)
- Volume anormal: >3x a media CONTRA sua posicao
- Invalidacao total: a tese que motivou a entrada nao e mais valida por multiplos fatores

=== FORMATO DE RESPOSTA (JSON) ===

Responda SEMPRE em JSON valido com esta estrutura:
{{
    "action": "LONG" | "SHORT" | "HOLD" | "CLOSE",
    "confidence": 0.0 a 1.0,
    "entry": preco_de_entrada (null se HOLD/CLOSE),
    "stop_loss": preco_do_sl (null se HOLD/CLOSE),
    "take_profit": preco_do_tp (null se HOLD/CLOSE),
    "confluence_count": numero de sinais alinhados (0-5),
    "signals_met": ["trend", "rsi", "macd", "volume", "bollinger"],
    "filters_blocked": ["adx_low", "atr_low", "rsi_extreme", "doji", "macro_divergence"],
    "reason": "explicacao curta do setup (max 2 frases)"
}}

Regras do JSON:
- confluence_count: quantos dos 5 sinais estao alinhados na direcao do trade
- signals_met: lista dos sinais que confirmam (apenas os que passaram)
- filters_blocked: lista de filtros que impediram (vazio [] se nenhum bloqueou)
- Se HOLD sem posicao: explique quais sinais faltam e quais filtros bloquearam
- Se HOLD com posicao: explique por que manter
- Se CLOSE: liste QUAIS indicadores confirmam a reversao
- Confianca minima para operar: {cfg.MIN_CONFIDENCE} (abaixo disso, retorne HOLD)

=== CONTEXTO DE SENTIMENTO (se disponivel) ===

Se dados de sentimento forem fornecidos (Fear & Greed Index, noticias, posts do X):
- Use como CONTEXTO adicional, NAO como sinal primario
- Fear & Greed 0-24 (Extreme Fear): mercado em panico, possivel oportunidade contrarian para LONG
- Fear & Greed 25-49 (Fear): sentimento negativo, cautela
- Fear & Greed 50 (Neutro): indecisao
- Fear & Greed 51-74 (Greed): sentimento positivo, mercado otimista
- Fear & Greed 75-100 (Extreme Greed): euforia, possivel oportunidade contrarian para SHORT
- Sentimento extremo CONTRA sua direcao = reduz confianca em 0.1
- Sentimento extremo A FAVOR (contrarian) = pode reforcar a tese
- NAO abra trades baseado APENAS em sentimento — exija confluencia tecnica

=== CALIBRACAO DE CONFIANCA ===

- 0.9-1.0: 5 confluencias + volume forte + tendencia diaria alinhada
- 0.8-0.9: 4 confluencias + sem filtro bloqueando
- 0.7-0.8: 3 confluencias claras
- 0.5-0.7: 2 confluencias ou sinais ambiguos = HOLD
- <0.5: setup fraco ou conflitante = HOLD obrigatorio

=== REGRA DE OURO ===

Em caso de duvida, retorne HOLD. Capital preservado e capital que opera amanha.
Nunca force um trade. Espere o setup vir ate voce."""


def create_analyst() -> "BaseAnalyst":
    """Factory: cria o analyst correto baseado no LLM_PROVIDER."""
    provider = cfg.LLM_PROVIDER.lower()
    if provider == "anthropic":
        return AnthropicAnalyst()
    elif provider == "google":
        return GoogleAnalyst()
    elif provider == "openai":
        return OpenAIAnalyst()
    elif provider == "xai":
        return XAIAnalyst()
    else:
        raise ValueError(f"Provider desconhecido: {cfg.LLM_PROVIDER}. Use: anthropic, google, openai, xai")


class BaseAnalyst:
    """Base class para todos os analysts."""

    def __init__(self, provider_name: str, model: str):
        self.provider_name = provider_name
        self.model = model
        self.trade_history: list[dict] = []

    def analyze(self, market_data: str) -> dict:
        """Envia dados a LLM e retorna decisao."""
        user_msg = market_data + self._history_text()
        text = None

        try:
            text, input_tokens, output_tokens = self._call_llm(user_msg)

            # Extrai JSON da resposta
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            decision = self._parse_json(text)

            # Garante que confidence e numerico
            raw_conf = decision.get("confidence", 0)
            try:
                decision["confidence"] = max(0.0, min(1.0, float(raw_conf)))
            except (TypeError, ValueError):
                logger.warning(f"[{self.provider_name}] Confidence nao-numerico: {raw_conf}, usando 0.0")
                decision["confidence"] = 0.0

            # Validacao de action
            if decision["action"] not in ("LONG", "SHORT", "HOLD", "CLOSE"):
                logger.warning(f"[{self.provider_name}] Acao invalida: {decision['action']}")
                decision["action"] = "HOLD"

            if decision["confidence"] < cfg.MIN_CONFIDENCE and decision["action"] in ("LONG", "SHORT"):
                logger.info(f"[{self.provider_name}] Confianca baixa ({decision['confidence']:.1f}), convertendo para HOLD")
                decision["action"] = "HOLD"

            # Validacao de confluencia minima
            confluence = decision.get("confluence_count", 0)
            if decision["action"] in ("LONG", "SHORT") and confluence < 3:
                logger.info(f"[{self.provider_name}] Confluencia insuficiente ({confluence}/5), convertendo para HOLD")
                decision["action"] = "HOLD"

            # Validacao de entry/SL/TP para LONG/SHORT
            if decision["action"] in ("LONG", "SHORT"):
                entry = decision.get("entry")
                sl = decision.get("stop_loss")
                tp = decision.get("take_profit")

                if entry is None or sl is None or tp is None:
                    logger.warning(f"[{self.provider_name}] Entry/SL/TP ausentes, convertendo para HOLD")
                    decision["action"] = "HOLD"
                elif not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                             for v in (entry, sl, tp)):
                    logger.warning(f"[{self.provider_name}] Entry/SL/TP invalidos (nao-finito ou <= 0): "
                                   f"entry={entry} sl={sl} tp={tp}")
                    decision["action"] = "HOLD"
                elif decision["action"] == "LONG" and (sl >= entry or tp <= entry):
                    logger.warning(f"[{self.provider_name}] SL/TP invalidos para LONG: SL={sl} Entry={entry} TP={tp}")
                    decision["action"] = "HOLD"
                elif decision["action"] == "SHORT" and (sl <= entry or tp >= entry):
                    logger.warning(f"[{self.provider_name}] SL/TP invalidos para SHORT: SL={sl} Entry={entry} TP={tp}")
                    decision["action"] = "HOLD"

            # Log detalhado
            signals = decision.get("signals_met", [])
            filters = decision.get("filters_blocked", [])
            logger.info(
                f"[{self.provider_name}] {self.model} | {decision['action']} | "
                f"conf={decision['confidence']:.1f} | confluence={confluence}/5 | "
                f"signals={signals} | filters={filters} | "
                f"{decision.get('reason', '')}"
            )
            logger.info(f"[{self.provider_name}] Tokens: {input_tokens}in/{output_tokens}out")

            return decision

        except json.JSONDecodeError as e:
            resp_text = text if text is not None else "(sem resposta)"
            logger.error(f"[{self.provider_name}] JSON invalido: {e}\nResposta: {resp_text}")
            return {"action": "HOLD", "confidence": 0, "reason": f"Erro parsing JSON: {e}"}
        except Exception as e:
            logger.error(f"[{self.provider_name}] Erro: {e}")
            return {"action": "HOLD", "confidence": 0, "reason": f"Erro: {e}"}

    def _parse_json(self, text: str) -> dict:
        """Tenta parsear JSON, com fallback para JSON truncado."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Tenta recuperar JSON truncado (ex: falta fechar "reason" e "}")
            action = re.search(r'"action"\s*:\s*"(\w+)"', text)
            confidence = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            entry = re.search(r'"entry"\s*:\s*([\d.]+|null)', text)
            sl = re.search(r'"stop_loss"\s*:\s*([\d.]+|null)', text)
            tp = re.search(r'"take_profit"\s*:\s*([\d.]+|null)', text)
            confluence = re.search(r'"confluence_count"\s*:\s*(\d+)', text)
            reason = re.search(r'"reason"\s*:\s*"([^"]*)', text)

            if action:
                result = {
                    "action": action.group(1),
                    "confidence": float(confidence.group(1)) if confidence else 0,
                    "entry": float(entry.group(1)) if entry and entry.group(1) != "null" else None,
                    "stop_loss": float(sl.group(1)) if sl and sl.group(1) != "null" else None,
                    "take_profit": float(tp.group(1)) if tp and tp.group(1) != "null" else None,
                    "confluence_count": int(confluence.group(1)) if confluence else 0,
                    "signals_met": [],
                    "filters_blocked": [],
                    "reason": reason.group(1) if reason else "JSON truncado",
                }
                logger.warning(f"[{self.provider_name}] JSON truncado recuperado: {result['action']}")
                return result
            raise

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        """Chama a LLM. Retorna (text, input_tokens, output_tokens). Override nos filhos."""
        raise NotImplementedError

    def _history_text(self) -> str:
        if not self.trade_history:
            return ""
        recent = self.trade_history[-10:]
        lines = ["\n\n=== HISTORICO RECENTE DE DECISOES ==="]
        for h in recent:
            conf = h.get('confidence', 0)
            try:
                conf_str = f"{float(conf):.1f}"
            except (TypeError, ValueError):
                conf_str = str(conf)
            confluence = h.get('confluence_count', '?')
            lines.append(f"  {h['time']} | {h['action']} | conf={conf_str} | confluence={confluence}/5 | {h['reason']}")
            if h.get('result'):
                lines.append(f"    -> Resultado: {h['result']}")
        return "\n".join(lines)

    def record_decision(self, decision: dict, timestamp: str, result: str = None):
        self.trade_history.append({
            "time": timestamp,
            "action": decision["action"],
            "confidence": decision.get("confidence", 0),
            "confluence_count": decision.get("confluence_count", 0),
            "reason": decision.get("reason", ""),
            "result": result,
        })
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]


class AnthropicAnalyst(BaseAnalyst):
    """Anthropic Claude."""

    def __init__(self):
        super().__init__("CLAUDE", cfg.ANTHROPIC_MODEL)
        import anthropic
        self.client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,  # Baixa temperatura para consistencia
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": user_msg}]
        )
        if not response.content:
            raise ValueError("Resposta vazia da API Anthropic")
        text = response.content[0].text.strip()
        return text, response.usage.input_tokens, response.usage.output_tokens


class GoogleAnalyst(BaseAnalyst):
    """Google Gemini."""

    def __init__(self):
        super().__init__("GEMINI", cfg.GOOGLE_MODEL)
        from google import genai
        self.client = genai.Client(api_key=cfg.GOOGLE_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=_build_system_prompt(),
                max_output_tokens=2048,
                temperature=0.1,  # Baixa temperatura para consistencia
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        if not response.text:
            raise ValueError("Resposta vazia da API Google")
        text = response.text.strip()
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        return text, input_tokens, output_tokens


class OpenAIAnalyst(BaseAnalyst):
    """OpenAI GPT."""

    def __init__(self):
        super().__init__("OPENAI", cfg.OPENAI_MODEL)
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.OPENAI_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,  # Baixa temperatura para consistencia
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user_msg},
            ]
        )
        if not response.choices:
            raise ValueError("Resposta vazia da API OpenAI")
        text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens


class XAIAnalyst(BaseAnalyst):
    """xAI Grok — suporta x_search e web_search para sentimento em tempo real."""

    def __init__(self):
        super().__init__("GROK", cfg.XAI_MODEL)
        self.use_search = cfg.XAI_SEARCH
        if self.use_search:
            self._init_responses_client()
        else:
            self._init_chat_client()

    def _init_chat_client(self):
        """Client OpenAI-compatible para chat completions (sem search)."""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=cfg.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    def _init_responses_client(self):
        """Prepara headers para Responses API (com search tools)."""
        self._api_key = cfg.XAI_API_KEY
        self._base_url = "https://api.x.ai/v1"

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        if self.use_search:
            return self._call_with_search(user_msg)
        return self._call_chat(user_msg)

    def _call_chat(self, user_msg: str) -> tuple[str, int, int]:
        """Chat completions padrao (sem search)."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.1,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user_msg},
            ]
        )
        if not response.choices:
            raise ValueError("Resposta vazia da API xAI")
        text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens

    def _call_with_search(self, user_msg: str) -> tuple[str, int, int]:
        """Responses API com x_search + web_search para sentimento em tempo real."""
        import requests as req

        search_prompt = (
            "Antes de analisar os indicadores tecnicos, busque informacoes recentes sobre Bitcoin: "
            "1) Sentimento atual no X (Twitter) sobre BTC — bullish ou bearish? "
            "2) Noticias recentes que podem impactar o preco do Bitcoin. "
            "3) Nivel de medo/ganancia do mercado crypto. "
            "Use essas informacoes como CONTEXTO adicional junto com os indicadores tecnicos abaixo.\n\n"
        )

        payload = {
            "model": self.model,
            "instructions": _build_system_prompt(),
            "input": search_prompt + user_msg,
            "tools": [
                {"type": "x_search"},
                {"type": "web_search"},
            ],
            "temperature": 0.1,
            "max_output_tokens": 2048,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        resp = req.post(
            f"{self._base_url}/responses",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extrair texto da resposta (pode ter multiplos output items)
        text = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        text += content.get("text", "")

        if not text:
            raise ValueError("Resposta vazia da API xAI Responses")

        text = text.strip()
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return text, input_tokens, output_tokens
