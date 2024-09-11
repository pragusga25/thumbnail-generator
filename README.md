# Dynamic Video Thumbnail Generator

This project is a Python application that generates dynamic thumbnails from video URLs. It uses computer vision techniques to select key frames and create an animated GIF thumbnail.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/pragusga25/thumbnail-generator.git
   cd thumbnail-generator
   ```

2. Create a virtual environment:

   - On Windows:

     ```
     python -m venv venv
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```
     python3 -m venv venv
     source venv/bin/activate
     ```

3. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the FastAPI server:

   ```
   uvicorn main:app --reload
   ```

2. The server will start running on `http://127.0.0.1:8000`

3. You can now send POST requests to `http://127.0.0.1:8000/generate` with a JSON body containing the video URL:

   ```json
   {
     "url": "https://www.w3schools.com/html/mov_bbb.mp4"
   }
   ```

4. The server will respond with a GIF thumbnail of the video.

## Running Tests

To run the tests, make sure you're in the project directory and your virtual environment is activated. Then run:

```
pytest
```

This will discover and run all the tests in the `test_main.py` file. The tests include both synchronous and asynchronous tests, which are handled automatically by pytest-asyncio.

## Frame Selection Algorithm

The application uses an advanced algorithm to select key frames for the thumbnail:

1. **Frame Difference Calculation**: The algorithm computes the difference between consecutive frames to identify significant changes in the video content.

2. **Peak Detection**: Using the scipy library's `find_peaks` function, the algorithm identifies peaks in the frame differences. These peaks likely represent scene changes or moments of high activity in the video.

3. **Frame Selection**:

   - If there are fewer peaks than the desired number of frames, additional frames with the highest differences are included.
   - If there are more peaks than needed, the algorithm selects the peaks with the highest difference scores.

4. **Thumbnail Creation**: The selected frames are then used to create an animated GIF thumbnail.

This approach ensures that the thumbnail captures the most representative and visually interesting moments from the video, providing a good overview of its content.

## Notes

- Ensure you have sufficient disk space, as the application temporarily downloads videos to process them.
- The application may require additional system libraries for video processing. If you encounter any errors, please refer to the OpenCV and MoviePy documentation for system-specific requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
