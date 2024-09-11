import pytest
from fastapi.testclient import TestClient
from main import (
    app,
    download_video,
    extract_key_frames,
    create_animated_thumbnail,
)
import tempfile
import os
import numpy as np
import cv2
from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send

client = TestClient(app)


@pytest.mark.asyncio
async def test_download_video(monkeypatch):
    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.content = b"fake video content"

            def iter_content(self, chunk_size):
                yield self.content

            def raise_for_status(self):
                pass

        return MockResponse()

    monkeypatch.setattr("requests.get", mock_get)

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = await download_video("http://fake-url.com/video.mp4")
        assert os.path.exists(video_path)
        with open(video_path, "rb") as f:
            assert f.read() == b"fake video content"


def test_extract_key_frames(monkeypatch):
    def mock_VideoCapture(*args, **kwargs):
        class MockVideoCapture:
            def get(self, prop):
                return 10 if prop == cv2.CAP_PROP_FRAME_COUNT else 30

            def read(self):
                return True, np.zeros((100, 100, 3), dtype=np.uint8)

            def set(self, *args):
                pass

            def release(self):
                pass

        return MockVideoCapture()

    monkeypatch.setattr("cv2.VideoCapture", mock_VideoCapture)

    frames = extract_key_frames("fake_path.mp4", num_frames=5)
    assert len(frames) == 5
    assert all(frame.shape == (100, 100, 3) for frame in frames)


def test_create_animated_thumbnail():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    thumbnail_path = create_animated_thumbnail(frames)
    assert os.path.exists(thumbnail_path)
    assert thumbnail_path.endswith(".gif")


@pytest.mark.asyncio
async def test_generate_thumbnail_endpoint(monkeypatch):
    async def mock_download_video(*args, **kwargs):
        return "fake_video_path.mp4"

    def mock_extract_key_frames(*args, **kwargs):
        return [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]

    def mock_create_animated_thumbnail(*args, **kwargs):
        return "fake_thumbnail.gif"

    def mock_unlink(*args, **kwargs):
        pass  # Do nothing when trying to delete the file

    async def mock_call(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [[b"content-type", b"image/gif"]],
            }
        )

    monkeypatch.setattr("main.download_video", mock_download_video)
    monkeypatch.setattr("main.extract_key_frames", mock_extract_key_frames)
    monkeypatch.setattr(
        "main.create_animated_thumbnail", mock_create_animated_thumbnail
    )
    monkeypatch.setattr("os.unlink", mock_unlink)

    # Mock FileResponse
    monkeypatch.setattr(FileResponse, "__call__", mock_call)

    response = client.post("/generate", json={"url": "http://fake-url.com/video.mp4"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/gif"


if __name__ == "__main__":
    pytest.main()
