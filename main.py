#!/usr/bin/env python3
"""
Autonomous YouTube Live streaming tool.

This script composes a video source and shuffled audio tracks into a single
RTMP stream and pushes it to YouTube Live. The configuration is read from a
`.env` file that must provide the stream URL and key. Optional Discord webhook
notifications can report status updates.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
import psutil  # type: ignore


def parse_int(value: Optional[str]) -> int:
    try:
        return int(value or "0")
    except (TypeError, ValueError):
        return 0


def parse_float(value: Optional[str]) -> float:
    try:
        return float(value or "0")
    except (TypeError, ValueError):
        return 0.0


def print_environment_info() -> None:
    """Print debugging information about the current Python environment."""
    executable = Path(sys.executable)
    print("[env] Python executable:", executable)
    print("[env] Python version:", sys.version.replace("\n", " "))
    print("[env] Python bin directory:", executable.parent)
    print("[env] Current working directory:", Path.cwd())


# -------------------------------------------------------------------------------------------------
# Configuration and status structures
# -------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    stream_url: str
    stream_key: str
    discord_webhook: Optional[str]
    base_dir: Path
    audio_dir: Path
    display_dir: Path
    ffmpeg_path: str
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    track_gap_seconds: float = 2.0

    @classmethod
    def from_environment(cls) -> "Config":
        base_dir = Path.cwd()
        stream_url = os.environ.get("YOUTUBE_STREAM_URL")
        stream_key = os.environ.get("YOUTUBE_STREAM_KEY")
        if not stream_url or not stream_key:
            raise RuntimeError(
                "Missing YOUTUBE_STREAM_URL or YOUTUBE_STREAM_KEY in environment."
            )

        discord_webhook_raw = os.environ.get("DISCORD_WEBHOOK_URL") or ""
        discord_webhook = discord_webhook_raw.strip() or None
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise RuntimeError("ffmpeg executable not found in PATH.")
        return cls(
            stream_url=stream_url.rstrip("/"),
            stream_key=stream_key.strip(),
            discord_webhook=discord_webhook,
            base_dir=base_dir,
            audio_dir=base_dir / "audio",
            display_dir=base_dir / "display",
            ffmpeg_path=ffmpeg_path,
        )

    @property
    def prepared_static_video_path(self) -> Path:
        return self.display_dir / "image_prepared.mp4"


# -------------------------------------------------------------------------------------------------
# Discord notifier
# -------------------------------------------------------------------------------------------------


class DiscordNotifier:
    def __init__(self, webhook_url: Optional[str], stop_event: threading.Event) -> None:
        self.webhook_url = webhook_url
        self.stop_event = stop_event
        self.daily_thread: Optional[threading.Thread] = None

    def post(self, message: str) -> None:
        if not self.webhook_url:
            return
        payload = {"content": message}
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            self.webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "youtube-streamer/1.0 (https://discord.com)",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=10):
                pass
        except HTTPError as exc:  # pragma: no cover - network failure
            body = exc.read().decode("utf-8", errors="ignore")
            print(
                f"[discord] failed to post message: {exc.code} {exc.reason} body={body.strip()}"
            )
        except URLError as exc:  # pragma: no cover - network failure
            print(f"[discord] failed to post message: {exc}")

    def start_daily_reporting(
        self, status_provider: Callable[[], str], first_run_delay: int = 10
    ) -> None:
        if not self.webhook_url:
            return
        if self.daily_thread:
            return

        def _loop() -> None:
            # Initial small delay to allow metrics to populate.
            if not self._wait(first_run_delay):
                return
            while not self.stop_event.is_set():
                summary = status_provider()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.post(f"[Daily Status] {timestamp}\n{summary}")
                # Wait roughly 24h for the next status, but break if stopped.
                if not self._wait(24 * 3600):
                    break

        self.daily_thread = threading.Thread(
            target=_loop, name="discord-daily", daemon=True
        )
        self.daily_thread.start()

    def _wait(self, seconds: float) -> bool:
        """Wait for given seconds or until stop_event set."""
        deadline = time.time() + seconds
        while not self.stop_event.is_set() and time.time() < deadline:
            time.sleep(1)
        return not self.stop_event.is_set()


# -------------------------------------------------------------------------------------------------
# Metrics handling
# -------------------------------------------------------------------------------------------------


@dataclass
class MetricsSnapshot:
    bitrate_mbps: float
    fps: float
    drop_frames: int
    total_frames: int
    out_time_seconds: float
    cpu_percent: float
    memory_mb: float
    current_track: Optional[str]


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._progress: Dict[str, str] = {}
        self._last_console_at = 0.0
        self._current_track: Optional[str] = None

    def update_progress(self, key: str, value: str) -> None:
        with self._lock:
            self._progress[key] = value

    def update_current_track(self, track: Optional[str]) -> None:
        with self._lock:
            self._current_track = track

    def snapshot(self) -> MetricsSnapshot:
        with self._lock:
            bitrate_str = self._progress.get("bitrate", "")
            bitrate_mbps = 0.0
            if bitrate_str and bitrate_str != "N/A":
                try:
                    if bitrate_str.endswith("kbits/s"):
                        bitrate_mbps = float(bitrate_str.replace("kbits/s", "").strip()) / 1000.0
                    else:
                        bitrate_mbps = float(bitrate_str) / 1_000_000.0
                except ValueError:
                    bitrate_mbps = 0.0

            total_size = parse_int(self._progress.get("total_size"))
            out_time_ms = parse_int(self._progress.get("out_time_ms"))
            fps = parse_float(self._progress.get("fps"))
            drop_frames = parse_int(self._progress.get("drop_frames"))
            total_frames = parse_int(self._progress.get("frame"))
            if bitrate_mbps == 0.0:
                bitrate_mbps = self._compute_mbps_from_size(total_size, out_time_ms)

            cpu_percent, memory_mb = get_process_resource_usage()

            return MetricsSnapshot(
                bitrate_mbps=bitrate_mbps,
                fps=fps,
                drop_frames=drop_frames,
                total_frames=total_frames,
                out_time_seconds=out_time_ms / 1000.0,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                current_track=self._current_track,
            )

    def should_print(self, interval_seconds: float) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_console_at >= interval_seconds:
                self._last_console_at = now
                return True
            return False

    @staticmethod
    def _compute_mbps_from_size(total_size_bytes: int, out_time_ms: int) -> float:
        if out_time_ms <= 0:
            return 0.0
        bits = total_size_bytes * 8.0
        seconds = out_time_ms / 1000.0
        return bits / 1_000_000.0 / seconds if seconds else 0.0


# -------------------------------------------------------------------------------------------------
# Resource usage helpers
# -------------------------------------------------------------------------------------------------


def get_process_resource_usage() -> Tuple[float, float]:
    """Return (cpu_percent, memory_mb) using psutil; fall back to zeros on failure."""
    try:
        proc = psutil.Process()
        cpu = proc.cpu_percent(interval=None)
        mem = proc.memory_info().rss / (1024 * 1024)
        return cpu, mem
    except Exception:
        return 0.0, 0.0


# -------------------------------------------------------------------------------------------------
# Audio streaming thread
# -------------------------------------------------------------------------------------------------


class AudioFeeder(threading.Thread):
    """Decode MP3 tracks into raw PCM and feed them into ffmpeg stdin."""

    def __init__(
        self,
        config: Config,
        metrics: MetricsStore,
        ffmpeg_stdin,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="audio-feeder", daemon=True)
        self.config = config
        self.metrics = metrics
        self.ffmpeg_stdin = ffmpeg_stdin
        self.stop_event = stop_event
        self.chunk_size = 4096

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                playlist = self._build_playlist()
                if not playlist:
                    if not self._emit_silence(duration_seconds=1):
                        break
                    continue
                for track in playlist:
                    if self.stop_event.is_set():
                        break
                    self.metrics.update_current_track(track.name)
                    if not self._stream_track(track):
                        return
                    self.metrics.update_current_track(None)
                    if not self._sleep_between_tracks():
                        return
        finally:
            self.metrics.update_current_track(None)
            try:
                self.ffmpeg_stdin.close()
            except Exception:
                pass

    def _build_playlist(self) -> List[Path]:
        audio_files = [p for p in self.config.audio_dir.glob("*.mp3") if p.is_file()]
        if not audio_files:
            return []
        random.shuffle(audio_files)
        return audio_files

    def _stream_track(self, track: Path) -> bool:
        cmd = [
            self.config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(track),
            "-f",
            "s16le",
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "pipe:1",
        ]
        try:
            with subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as proc:
                while not self.stop_event.is_set():
                    chunk = proc.stdout.read(self.chunk_size)
                    if not chunk:
                        break
                    try:
                        self.ffmpeg_stdin.write(chunk)
                        self.ffmpeg_stdin.flush()
                    except BrokenPipeError:
                        return False
                proc.wait()
        except FileNotFoundError:
            print(f"[audio] missing ffmpeg while decoding {track}")
        except Exception as exc:
            print(f"[audio] failed to stream {track.name}: {exc}")
        return True

    def _emit_silence(self, duration_seconds: float) -> bool:
        bytes_per_second = (
            self.config.audio_sample_rate * self.config.audio_channels * 2
        )
        total_bytes = int(duration_seconds * bytes_per_second)
        if total_bytes <= 0:
            return True

        silent_chunk = b"\x00" * min(bytes_per_second, 65536)
        remaining = total_bytes
        while remaining > 0 and not self.stop_event.is_set():
            to_write = silent_chunk[: min(len(silent_chunk), remaining)]
            try:
                self.ffmpeg_stdin.write(to_write)
                self.ffmpeg_stdin.flush()
            except BrokenPipeError:
                return False
            remaining -= len(to_write)
            time.sleep(len(to_write) / bytes_per_second)
        return True

    def _sleep_between_tracks(self) -> bool:
        """Pause between tracks; return False if stopped during the wait."""
        delay = self.config.track_gap_seconds
        if delay <= 0:
            return True
        return self._emit_silence(delay)


# -------------------------------------------------------------------------------------------------
# FFmpeg process runner
# -------------------------------------------------------------------------------------------------


class FFmpegRunner:
    def __init__(
        self,
        config: Config,
        metrics: MetricsStore,
        notifier: DiscordNotifier,
        stop_event: threading.Event,
    ) -> None:
        self.config = config
        self.metrics = metrics
        self.notifier = notifier
        self.stop_event = stop_event

    def run(self) -> None:
        print("[main] starting streaming loop. Press Ctrl+C to stop.")
        while not self.stop_event.is_set():
            try:
                self._ensure_directories()
                cmd, video_mode = self._build_ffmpeg_command()
                print(f"[ffmpeg] launching (mode={video_mode})")
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
            except Exception as exc:
                self.notifier.post(f":warning: Failed to launch ffmpeg: {exc}")
                print(f"[ffmpeg] launch error: {exc}")
                self._sleep_with_stop(5)
                continue

            audio_thread = AudioFeeder(
                self.config,
                self.metrics,
                process.stdin,
                self.stop_event,
            )
            audio_thread.start()
            progress_thread = threading.Thread(
                target=self._consume_progress,
                args=(process,),
                name="ffmpeg-progress",
                daemon=True,
            )
            progress_thread.start()

            return_code = None
            try:
                while return_code is None:
                    if self.stop_event.is_set():
                        process.terminate()
                        break
                    return_code = process.poll()
                    time.sleep(0.5)
            except KeyboardInterrupt:
                self.stop_event.set()
                process.terminate()
            finally:
                audio_thread.join(timeout=2)
                if process.poll() is None:
                    process.kill()
                progress_thread.join(timeout=2)

            if self.stop_event.is_set():
                break

            if return_code == 0:
                print("[ffmpeg] exited cleanly, restarting to maintain stream.")
            else:
                message = (
                    f":x: ffmpeg exited with code {return_code}. Retrying in 5 seconds."
                )
                print(f"[ffmpeg] {message}")
                self.notifier.post(message)
                self._sleep_with_stop(5)

        print("[main] streaming loop stopped.")

    def _ensure_directories(self) -> None:
        for path in (self.config.audio_dir, self.config.display_dir):
            path.mkdir(parents=True, exist_ok=True)

    def _resolve_display_image(self) -> Optional[Path]:
        for name in ("image.jpg", "image.jpeg", "image.png"):
            candidate = self.config.display_dir / name
            if candidate.exists():
                return candidate
        return None

    def _prepare_static_video(
        self,
        filter_chain: str,
        image_path: Optional[Path],
        duration: int,
        target_fps: int,
        target_bitrate: str,
        target_bufsize: str,
    ) -> Optional[Path]:
        """Transcode the display image once so streaming can use copy mode."""
        if not image_path:
            return None

        target = self.config.prepared_static_video_path
        try:
            if target.exists() and target.stat().st_mtime >= image_path.stat().st_mtime:
                return target
        except OSError as exc:
            print(f"[ffmpeg] failed to stat prepared static video: {exc}")

        tmp_target = target.with_name(f"{target.stem}.tmp{target.suffix}")
        if tmp_target.exists():
            try:
                tmp_target.unlink()
            except OSError:
                pass

        keyint = max(target_fps, target_fps * 2)

        cmd = [
            self.config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-vf",
            filter_chain,
            "-r",
            str(target_fps),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            target_bitrate,
            "-maxrate",
            target_bitrate,
            "-bufsize",
            target_bufsize,
            "-g",
            str(keyint),
            "-keyint_min",
            str(keyint),
            "-sc_threshold",
            "0",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp_target),
        ]

        try:
            print("[ffmpeg] preparing optimized static video asset (one-time encode).")
            subprocess.run(cmd, check=True)
            tmp_target.replace(target)
            return target
        except subprocess.CalledProcessError as exc:
            print(f"[ffmpeg] failed to prepare static video: {exc}")
        except Exception as exc:
            print(f"[ffmpeg] unexpected error preparing static video: {exc}")

        try:
            if tmp_target.exists():
                tmp_target.unlink()
        except OSError:
            pass
        return None

    def _build_ffmpeg_command(self) -> Tuple[List[str], str]:
        display_image = self._resolve_display_image()
        filter_chain = (
            "scale=854:480:force_original_aspect_ratio=decrease,"
            "pad=854:480:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1"
        )

        target_fps = 30

        prepared_video = self._prepare_static_video(
            filter_chain,
            display_image,
            duration=120,
            target_fps=target_fps,
            target_bitrate="80k",
            target_bufsize="160k",
        )

        base_cmd = [
            self.config.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-progress",
            "pipe:2",
            "-nostdin",
        ]

        keyint = max(target_fps, target_fps * 2)

        reencode_video_args = [
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "80k",
            "-maxrate",
            "80k",
            "-bufsize",
            "160k",
            "-g",
            str(keyint),
            "-keyint_min",
            str(keyint),
            "-r",
            str(target_fps),
        ]

        if prepared_video:
            video_input = ["-re", "-stream_loop", "-1", "-i", str(prepared_video)]
            video_mode = "image_copy"
            video_output_args = ["-c:v", "copy"]
        elif display_image:
            video_input = [
                "-loop",
                "1",
                "-framerate",
                str(target_fps),
                "-i",
                str(display_image),
            ]
            video_mode = "image_encode"
            video_output_args = list(reencode_video_args)
        else:
            video_input = [
                "-f",
                "lavfi",
                "-re",
                "-i",
                f"color=c=black:s=854x480:r={target_fps}",
            ]
            video_mode = "color"
            video_output_args = list(reencode_video_args)

        audio_input = [
            "-f",
            "s16le",
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "-i",
            "pipe:0",
        ]

        output_url = f"{self.config.stream_url}/{self.config.stream_key}"
        output_args = video_output_args + [
            "-c:a",
            "aac",
            "-b:a",
            "64k",
            "-ar",
            str(self.config.audio_sample_rate),
            "-ac",
            str(self.config.audio_channels),
            "-f",
            "flv",
            output_url,
        ]

        return base_cmd + video_input + audio_input + output_args, video_mode

    def _consume_progress(self, process: subprocess.Popen) -> None:
        if not process.stderr:
            return
        while not self.stop_event.is_set():
            line = process.stderr.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if "=" not in decoded:
                continue
            key, value = decoded.split("=", 1)
            self.metrics.update_progress(key.strip(), value.strip())
            if key == "progress" and value.strip() == "end":
                break
            if self.metrics.should_print(15):
                snapshot = self.metrics.snapshot()
                track = snapshot.current_track or "None"
                print(
                    f"[progress] bitrate={snapshot.bitrate_mbps:.2f}Mbps "
                    f"fps={snapshot.fps:.1f} drop_frames={snapshot.drop_frames} "
                    f"time={snapshot.out_time_seconds:.1f}s "
                    f"cpu={snapshot.cpu_percent:.1f}% mem={snapshot.memory_mb:.1f}MB "
                    f"track={track}"
                )

    def _sleep_with_stop(self, seconds: float) -> None:
        deadline = time.time() + seconds
        while not self.stop_event.is_set() and time.time() < deadline:
            time.sleep(0.5)


# -------------------------------------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------------------------------------


def main() -> None:
    print_environment_info()
    load_dotenv(dotenv_path=Path(".env"), override=False)
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    try:
        config = Config.from_environment()
    except Exception as exc:
        print(f"[config] {exc}")
        raise SystemExit(1)

    metrics = MetricsStore()
    notifier = DiscordNotifier(config.discord_webhook, stop_event)
    notifier.post(
        f":rocket: Streamer starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
    )
    notifier.start_daily_reporting(lambda: format_metrics(metrics))

    runner = FFmpegRunner(config, metrics, notifier, stop_event)
    try:
        runner.run()
    finally:
        stop_event.set()
        notifier.post(":white_check_mark: Streamer stopped.")


def format_metrics(metrics: MetricsStore) -> str:
    snapshot = metrics.snapshot()
    track = snapshot.current_track or "None"
    return (
        f"bitrate={snapshot.bitrate_mbps:.2f}Mbps, "
        f"fps={snapshot.fps:.1f}, drop_frames={snapshot.drop_frames}, "
        f"elapsed={snapshot.out_time_seconds/3600:.2f}h, "
        f"cpu={snapshot.cpu_percent:.1f}%, mem={snapshot.memory_mb:.1f}MB, "
        f"track={track}"
    )


if __name__ == "__main__":
    main()
