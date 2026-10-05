"""Camada de acesso ao modelo de linguagem (plugável).

Provedores (LLM_PROVIDER no .env):
- hf                : modelo local via transformers (padrão: Qwen/Qwen2.5-1.5B-Instruct).
                      Gratuito e offline após o download; lento em CPU.
- ollama            : servidor Ollama local (API compatível com OpenAI em
                      http://localhost:11434/v1). Ex.: LLM_MODEL=qwen2.5:3b
- openai            : API da OpenAI (OPENAI_API_KEY).
- gemini            : API do Google Gemini via endpoint compatível com OpenAI (GEMINI_API_KEY).
                      Ex.: LLM_MODEL=gemini-3.8-flash. Temperatura não é enviada (recomendação do Google).
- anthropic         : API da Anthropic (ANTHROPIC_API_KEY).
- openai_compatible : qualquer servidor compatível (LLM_BASE_URL + LLM_API_KEY).
- extractive        : SEM LLM. Lista as evidências agrupadas, sem síntese.
                      Usado em testes automatizados e como modo de contingência.

Chaves de API ficam APENAS no .env (que está no .gitignore).
Temperatura 0 por padrão quando o provedor aceita (ver AnthropicLLM para exceções).
"""
from __future__ import annotations

import os
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str: ...


class HFLocalLLM:
    def __init__(self, s: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf:{s.llm_model}"
        self.s = s
        self.tok = AutoTokenizer.from_pretrained(s.llm_model)
        cuda = torch.cuda.is_available()
        kwargs = {"device_map": "auto"} if cuda else {}
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}.get(
            s.llm_dtype.lower(), torch.float16 if cuda else torch.float32)
        try:  # transformers >= 4.56 usa `dtype`; versões anteriores, `torch_dtype`
            self.model = AutoModelForCausalLM.from_pretrained(s.llm_model, dtype=dtype, **kwargs)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(s.llm_model, torch_dtype=dtype, **kwargs)
        self.model.eval()
        torch.manual_seed(s.seed)

    def generate(self, system: str, user: str) -> str:
        import torch

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        do_sample = self.s.llm_temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.s.llm_max_new_tokens, do_sample=do_sample,
                temperature=self.s.llm_temperature if do_sample else None,
                top_p=None, top_k=None, pad_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    DEFAULT_URLS = {
        "ollama": "http://localhost:11434/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "openai": None,
    }
    KEY_VARS = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "ollama": None,
                "openai_compatible": "LLM_API_KEY"}

    def __init__(self, s: Settings, provider: str):
        from openai import OpenAI

        self.s = s
        self.provider = provider
        self.name = f"{provider}:{s.llm_model}"
        base_url = s.llm_base_url or self.DEFAULT_URLS.get(provider)
        key_var = self.KEY_VARS.get(provider)
        api_key = (os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")) if provider == "gemini" \
            else (os.getenv(key_var, "") if key_var else "ollama")
        if key_var and not api_key:
            raise RuntimeError(f"Defina {key_var} no arquivo .env para usar LLM_PROVIDER={provider}.")
        # Novas tentativas automáticas (com espera crescente) em erro 429/5xx:
        # útil no plano gratuito do Gemini, que limita requisições por minuto.
        self.client = OpenAI(api_key=api_key, base_url=base_url,
                             max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")))
        # Gemini 3: o Google recomenda manter a temperatura no padrão (1.0); valores
        # menores podem causar repetição/loop. O raciocínio ("thinking") conta como
        # saída, então max_tokens precisa de folga, e o nível é controlado por
        # reasoning_effort (minimal | low | medium | high).
        gemini = provider == "gemini"
        self.send_temperature = os.getenv("LLM_SEND_TEMPERATURE", "false" if gemini else "true").lower() in ("1", "true", "sim")
        self.max_tokens = int(os.getenv("LLM_API_MAX_TOKENS", "8000" if gemini else str(s.llm_max_new_tokens)))
        self.reasoning_effort = os.getenv("LLM_REASONING_EFFORT", "low" if gemini else "").strip()

    def generate(self, system: str, user: str) -> str:
        kwargs = dict(model=self.s.llm_model, max_tokens=self.max_tokens,
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        if self.send_temperature:
            kwargs["temperature"] = self.s.llm_temperature
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        r = self.client.chat.completions.create(**kwargs)
        text = (r.choices[0].message.content or "").strip()
        if not text and r.choices[0].finish_reason == "length":
            raise RuntimeError("A resposta foi cortada por max_tokens (o raciocínio consumiu o limite). "
                               "Aumente LLM_API_MAX_TOKENS no .env.")
        return text


class AnthropicLLM:
    """API da Anthropic.

    Modelos recentes (ex.: claude-opus-5-5) têm raciocínio ("thinking")
    sempre ligado e recusam temperature/top_p/top_k diferentes do padrão
    (erro 400). Por isso:
    - a temperatura NÃO é enviada (as respostas podem variar entre execuções;
      a verificação pós-geração continua garantindo a fundamentação);
    - max_tokens precisa comportar raciocínio + resposta (ANTHROPIC_MAX_TOKENS);
    - a profundidade do raciocínio é controlada por `effort` (ANTHROPIC_EFFORT:
      low | medium | high...). "low" basta para sintetizar avaliações curtas.
    Para modelos antigos que aceitam temperatura, defina ANTHROPIC_SEND_TEMPERATURE=true.
    """

    def __init__(self, s: Settings):
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Defina ANTHROPIC_API_KEY no arquivo .env para usar LLM_PROVIDER=anthropic.")
        self.s = s
        self.name = f"anthropic:{s.llm_model}"
        self.client = anthropic.Anthropic(max_retries=int(os.getenv("LLM_MAX_RETRIES", "6")))
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))
        self.effort = os.getenv("ANTHROPIC_EFFORT", "low").strip()
        self.send_temperature = os.getenv("ANTHROPIC_SEND_TEMPERATURE", "false").lower() in ("1", "true", "sim")

    def generate(self, system: str, user: str) -> str:
        kwargs = dict(model=self.s.llm_model, system=system, max_tokens=self.max_tokens,
                      messages=[{"role": "user", "content": user}])
        if self.send_temperature:
            kwargs["temperature"] = self.s.llm_temperature
        if self.effort:
            # extra_body funciona mesmo em versões do SDK sem o campo nomeado.
            kwargs["extra_body"] = {"output_config": {"effort": self.effort}}
        r = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        if not text and getattr(r, "stop_reason", "") == "max_tokens":
            raise RuntimeError("A resposta do modelo foi cortada por max_tokens (o raciocínio consumiu o limite). "
                               "Aumente ANTHROPIC_MAX_TOKENS no .env.")
        return text


class ExtractiveLLM:
    """Modo sem LLM: devolve as evidências citadas, sem síntese. Garante
    rastreabilidade total; serve para testes e contingência."""

    name = "extractive"

    def generate(self, system: str, user: str) -> str:
        import re

        labels = re.findall(r"^\[(E\d+)\][^\n]*\n\"([^\n]*)\"", user, flags=re.M)
        if not labels:
            from .prompts import SEM_EVIDENCIA_TOKEN

            return SEM_EVIDENCIA_TOKEN
        lines = [f"Resumo: modo extrativo (sem LLM) — trechos das avaliações recuperadas. [{labels[0][0]}]"]
        lines += [f"- \"{txt}\" [{lab}]" for lab, txt in labels]
        return "\n".join(lines)


def get_llm(settings: Settings) -> LLM:
    p = settings.llm_provider.lower()
    if p == "hf":
        return HFLocalLLM(settings)
    if p in ("ollama", "openai", "gemini", "openai_compatible"):
        return OpenAICompatibleLLM(settings, p)
    if p == "anthropic":
        return AnthropicLLM(settings)
    if p == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"LLM_PROVIDER desconhecido: {settings.llm_provider}")
