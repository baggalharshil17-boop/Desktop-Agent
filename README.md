# Financial Voice Agent

A read-only AI voice assistant for a Zerodha Kite trading dashboard. See
`Financial_Voice_Agent.claude.md` for the full design/developer reference.

## Getting Started (Fresh Clone)

### 1. Prerequisites

- Python 3.10+ (developed against 3.14, also verified on 3.10).
- `pip install -r requirements.txt` installs everything, but two packages
  need a callout on Windows:
  - **torch / torchaudio** (used for voice-activity detection via
    silero-vad): install **both together, from the CPU index, before
    anything else**. Plain `pip install torch` may pull a multi-GB CUDA
    build, and — more importantly — installing them separately or from
    different indexes gives you mismatched builds whose compiled extensions
    can't link, failing at startup with
    `OSError: [WinError 127] The specified procedure could not be found`:
    ```
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```
    Verify before continuing — both versions should print with `+cpu` and
    the extension should report `True`:
    ```
    python -c "import torch, torchaudio; from torchaudio._extension import _IS_TORCHAUDIO_EXT_AVAILABLE as e; print(torch.__version__, torchaudio.__version__, 'ext:', e)"
    ```
  - **pyaudio** (microphone/speaker access): sometimes fails to build from
    source on Windows. If `pip install -r requirements.txt` fails on it
    specifically, install a prebuilt wheel instead:
    ```
    pip install pipwin
    pipwin install pyaudio
    ```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the setup wizard

```
python scripts/setup.py
```

Walks you through picking providers and pasting in your own API keys,
validating each one live as you go. Writes `.env` and `config.yaml` for
you -- see the table below for what each provider needs.

### 4. Run it

```
python -m financial_voice_agent
```

Talk after it prints "Listening...". Ctrl+C to stop.

If you chose `mode: "live"` during setup, get today's Kite access token
first (it expires daily):
```
python scripts/kite_login.py
```

## What each provider needs

| Provider | Env var | Get it from | Notes |
|---|---|---|---|
| Groq (STT and/or LLM) | `GROQ_API_KEY` | console.groq.com | Free tier available |
| Hugging Face (STT and/or LLM alternative) | `HF_TOKEN` | huggingface.co/settings/tokens | Free tier available, separate quota from Groq |
| Cartesia (TTS) | `CARTESIA_API_KEY` | play.cartesia.ai | Credit-based free tier |
| Fish Audio (TTS alternative) | `FISH_AUDIO_API_KEY` | fish.audio/app/api-keys | `s2.1-pro-free` model works free; other models need funded API credit |
| Tavily (news search) | `TAVILY_API_KEY` | tavily.com | Free to 1k searches/month |
| Indian Stock API (fundamentals -- P/E, market cap, etc.) | `INDIAN_STOCK_API_KEY` | indianapi.in (dashboard's API/Manage Keys section) | Free plan available |
| Alpha Vantage (momentum stock screening) | `ALPHA_VANTAGE_API_KEY` | alphavantage.co/support/#api-key | Free tier: 5 requests/min, 25/day -- a single screen can take up to ~2 minutes due to built-in rate-limit pacing |
| Zerodha Kite Connect (live trading data) | `KITE_API_KEY` / `KITE_API_SECRET` / `KITE_ACCESS_TOKEN` | developers.kite.trade | Paid (~₹500/month), only needed for `mode: "live"` -- skip entirely with `mode: "mock"` |

## Development

Run tests: `python -m pytest -q`
