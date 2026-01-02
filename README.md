# opuscord

discord self-bot powered by claude opus 4.5 with semantic memory

![opuscord demo](media/image.png)

## run

```bash
uv sync
cp .env.example .env  # fill in your keys
uv run bot.py
```

## env

```
DISCORD_TOKEN=         # devtools -> network -> authorization header
TARGET_CHANNEL_ID=     # right click channel -> copy id
BOT_USER_ID=           # your bot account's user id
ANTHROPIC_API_KEY=     # console.anthropic.com
VOYAGE_API_KEY=        # dash.voyageai.com
```

## note

self-bot = discord tos violation. use at your own risk.
