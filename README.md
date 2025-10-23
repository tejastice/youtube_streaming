# Autonomous YouTube Streamer

Python tool that continuously streams shuffled audio tracks together with a looping video or fallback image to YouTube Live via RTMP. Environment variables provide the stream endpoint, credentials, and optional Discord webhook notifications.

## Prerequisites
- Python 3.11.x
- `ffmpeg` available on `PATH`
- RTMP stream URL/key for the YouTube Live event

## Local Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in your stream credentials
python main.py
```

## Deployment Notes
- `Procfile` declares a `worker` process that runs `python main.py`.
- `nixpacks.toml` instructs Railway/Nixpacks to install `ffmpeg`.
- Set the following environment variables in your hosting platform:
  - `YOUTUBE_STREAM_URL`
  - `YOUTUBE_STREAM_KEY`
  - `DISCORD_WEBHOOK_URL` (optional)

### Railway Workflow (GitHub Integration)
1. Push this directory to a new GitHub repository.
2. Create a Railway project with **Deploy from GitHub**, select your repository, and leave the default build settings (Nixpacks will detect Python).
3. In Railway → Variables, add the environment variables listed above.
4. Deploy; the `worker` service defined in the `Procfile` starts automatically and begins streaming.

Refer to `requirements.md` for the original functional specification.
