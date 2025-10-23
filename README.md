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
- `Procfile` declares a `worker`プロセスとして`python main.py`を実行します。
- RailwayでNixpacksを使う場合は`nixpacks.toml`が`ffmpeg`をインストールします（Dockerfileを使う場合は不要）。
- Set the following environment variables in your hosting platform:
  - `YOUTUBE_STREAM_URL`
  - `YOUTUBE_STREAM_KEY`
  - `DISCORD_WEBHOOK_URL` (optional)

### Railway Workflow (GitHub Integration)
1. Push this directory to a new GitHub repository.
2. Create a Railway project with **Deploy from GitHub**, select your repository, and leave the default build settings (Nixpacks will detect Python).
3. RailwayでDockerビルドを使う場合は、リポジトリ直下の`Dockerfile`が自動で選択され、イメージ内で`ffmpeg`をapt経由で導入します。Nixpacksを使いたい場合は`nixpacks.toml`を維持してください。
4. Railway → Variablesで上記環境変数を設定したらデプロイします。`Procfile`の`worker`（またはDockerのCMD）が起動して配信を開始します。

Refer to `requirements.md` for the original functional specification.
