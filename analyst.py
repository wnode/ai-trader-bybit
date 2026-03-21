"""
LLM Analyst — envia dados de mercado a uma LLM e recebe decisao de trade.
Suporta: Anthropic (Claude), Google (Gemini), OpenAI (GPT)
"""
import json
import re
import logging

from config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    GOOGLE_API_KEY, GOOGLE_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Voce e um trader profissional de Bitcoin Futures (BTCUSDT perpetual) na Bybit.
Voce opera com alavancagem 25x em timeframe de 15 minutos.

REGRAS:
1. Analise os dados de mercado fornecidos (candles, indicadores, contexto diario)
2. Decida: LONG, SHORT ou HOLD (nao operar agora)
3. Se LONG ou SHORT, defina entry, stop_loss e take_profit
4. SL deve ser entre 0.3% e 1.0% do preco de entrada
5. TP deve ser pelo menos 1.0x a distancia do SL (risk/reward >= 1:1)
6. Se ja existe posicao aberta, voce pode: HOLD (manter), CLOSE (fechar), ou ajustar SL/TP
7. Seja conservador — so entre quando ha alta confianca no setup
8. Considere: tendencia (EMAs), momentum (RSI, MACD), volatilidade (ATR, BB), volume
9. Evite operar em mercado lateral/ranging (ADX < 15)
10. Considere o contexto diario para nao operar contra a macro tendencia

RESPONDA SEMPRE em JSON valido com esta estrutura:
{
    "action": "LONG" | "SHORT" | "HOLD" | "CLOSE",
    "confidence": 0.0 a 1.0,
    "entry": preco_de_entrada (null se HOLD/CLOSE),
    "stop_loss": preco_do_sl (null se HOLD/CLOSE),
    "take_profit": preco_do_tp (null se HOLD/CLOSE),
    "reason": "explicacao curta do setup (max 2 frases)"
}

Se HOLD, explique por que nao ha setup.
Se CLOSE, explique por que fechar a posicao.
Confianca minima para operar: 0.7 (abaixo disso, retorne HOLD)."""


def create_analyst() -> "BaseAnalyst":
    """Factory: cria o analyst correto baseado no LLM_PROVIDER."""
    provider = LLM_PROVIDER.lower()
    if provider == "anthropic":
        return AnthropicAnalyst()
    elif provider == "google":
        return GoogleAnalyst()
    elif provider == "openai":
        return OpenAIAnalyst()
    else:
        raise ValueError(f"Provider desconhecido: {LLM_PROVIDER}. Use: anthropic, google, openai")


class BaseAnalyst:
    """Base class para todos os analysts."""

    def __init__(self, provider_name: str, model: str):
        self.provider_name = provider_name
        self.model = model
        self.trade_history: list[dict] = []

    def analyze(self, market_data: str) -> dict:
        """Envia dados a LLM e retorna decisao."""
        user_msg = market_data + self._history_text()

        try:
            text, input_tokens, output_tokens = self._call_llm(user_msg)

            # Extrai JSON da resposta
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            decision = self._parse_json(text)

            # Validacao
            if decision["action"] not in ("LONG", "SHORT", "HOLD", "CLOSE"):
                logger.warning(f"[{self.provider_name}] Acao invalida: {decision['action']}")
                decision["action"] = "HOLD"

            if decision.get("confidence", 0) < 0.7 and decision["action"] in ("LONG", "SHORT"):
                logger.info(f"[{self.provider_name}] Confianca baixa ({decision['confidence']:.1f}), convertendo para HOLD")
                decision["action"] = "HOLD"

            logger.info(f"[{self.provider_name}] {self.model} | {decision['action']} | conf={decision.get('confidence', 0):.1f} | {decision.get('reason', '')}")
            logger.info(f"[{self.provider_name}] Tokens: {input_tokens}in/{output_tokens}out")

            return decision

        except json.JSONDecodeError as e:
            logger.error(f"[{self.provider_name}] JSON invalido: {e}\nResposta: {text}")
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
            reason = re.search(r'"reason"\s*:\s*"([^"]*)', text)

            if action:
                result = {
                    "action": action.group(1),
                    "confidence": float(confidence.group(1)) if confidence else 0,
                    "entry": float(entry.group(1)) if entry and entry.group(1) != "null" else None,
                    "stop_loss": float(sl.group(1)) if sl and sl.group(1) != "null" else None,
                    "take_profit": float(tp.group(1)) if tp and tp.group(1) != "null" else None,
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
            lines.append(f"  {h['time']} | {h['action']} | conf={h['confidence']:.1f} | {h['reason']}")
            if h.get('result'):
                lines.append(f"    -> Resultado: {h['result']}")
        return "\n".join(lines)

    def record_decision(self, decision: dict, timestamp: str, result: str = None):
        self.trade_history.append({
            "time": timestamp,
            "action": decision["action"],
            "confidence": decision.get("confidence", 0),
            "reason": decision.get("reason", ""),
            "result": result,
        })
        if len(self.trade_history) > 50:
            self.trade_history = self.trade_history[-50:]


class AnthropicAnalyst(BaseAnalyst):
    """Anthropic Claude."""

    def __init__(self):
        super().__init__("CLAUDE", ANTHROPIC_MODEL)
        import anthropic
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        text = response.content[0].text.strip()
        return text, response.usage.input_tokens, response.usage.output_tokens


class GoogleAnalyst(BaseAnalyst):
    """Google Gemini."""

    def __init__(self):
        super().__init__("GEMINI", GOOGLE_MODEL)
        from google import genai
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=2048,
                temperature=0.3,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip()
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        return text, input_tokens, output_tokens


class OpenAIAnalyst(BaseAnalyst):
    """OpenAI GPT."""

    def __init__(self):
        super().__init__("OPENAI", OPENAI_MODEL)
        from openai import OpenAI
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _call_llm(self, user_msg: str) -> tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        text = response.choices[0].message.content.strip()
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return text, input_tokens, output_tokens
