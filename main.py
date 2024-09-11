from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import cv2
from moviepy.editor import *
import requests
import os
import tempfile
import numpy as np
from scipy.signal import find_peaks

app = FastAPI()


class VideoURL(BaseModel):
    url: str


async def download_video(url: str) -> str:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            return temp_file.name
    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to download video: {str(e)}"
        )


def extract_key_frames(video_path: str, num_frames: int = 10) -> list:
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    frame_differences = []
    prev_frame = None
    frame_indices = []

    for i in range(frame_count):
        ret, frame = video.read()
        if not ret:
            break

        frame_indices.append(i)

        if prev_frame is None:
            prev_frame = frame
            continue

        # Convert to grayscale for easier computation
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Compute frame difference
        frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
        diff_score = np.sum(frame_diff)
        frame_differences.append(diff_score)

        prev_frame = frame

    video.release()

    # Normalize frame differences
    frame_differences = np.array(frame_differences)
    max_min = np.max(frame_differences) - np.min(frame_differences)
    if max_min == 0:
        frame_differences = np.zeros_like(frame_differences)
    else:
        frame_differences = (frame_differences - np.min(frame_differences)) / max_min

    # Find peaks in frame differences (scene changes)
    peaks, _ = find_peaks(frame_differences, height=0.5, distance=fps)

    # If we don't have enough peaks, add frames with highest differences
    if len(peaks) < num_frames:
        additional_frames = num_frames - len(peaks)
        sorted_indices = np.argsort(frame_differences)[::-1]
        for idx in sorted_indices:
            if idx not in peaks:
                peaks = np.append(peaks, idx)
                additional_frames -= 1
                if additional_frames == 0:
                    break

    # If we have too many peaks, select the ones with highest differences
    if len(peaks) > num_frames:
        peak_values = frame_differences[peaks]
        top_indices = np.argsort(peak_values)[::-1][:num_frames]
        peaks = peaks[top_indices]

    # Sort peaks (frame indices) in ascending order
    peaks = sorted(peaks)

    # Extract the selected frames
    video = cv2.VideoCapture(video_path)
    frames = []
    for peak in peaks:
        video.set(cv2.CAP_PROP_POS_FRAMES, peak)
        ret, frame = video.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()

    return frames


def create_animated_thumbnail(frames: list) -> str:
    clip = ImageSequenceClip(frames, fps=1)
    output_path = tempfile.mktemp(suffix=".gif")
    clip.write_gif(output_path, fps=1)
    return output_path


@app.post("/generate")
async def generate_thumbnail(video: VideoURL):
    video_path = await download_video(video.url)

    def process_video():
        frames = extract_key_frames(video_path)
        thumbnail_path = create_animated_thumbnail(frames)
        os.unlink(video_path)  # Clean up the downloaded video
        return thumbnail_path

    thumbnail_path = await asyncio.to_thread(process_video)
    return FileResponse(
        thumbnail_path, media_type="image/gif", filename="thumbnail.gif"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
