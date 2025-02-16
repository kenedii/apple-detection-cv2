import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

# Import your apple detection function.
# It should be available in the apple_detection module.
from apple_detection import detect_apples

app = FastAPI(title="Apple Detector API")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Apple Detector API! Visit /docs for the API documentation."}

@app.post("/is_apple")
async def is_apple(file: UploadFile = File(...)):
    # Ensure the uploaded file is an image.
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    # Read the file into memory.
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Check that the image was correctly decoded.
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    # Use your apple detection function.
    processed_image, detected_apples = detect_apples(image)

    # Encode the processed image to JPEG format.
    success, encoded_image = cv2.imencode(".jpg", processed_image)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    # Convert the encoded image to bytes.
    image_bytes = io.BytesIO(encoded_image.tobytes())

    # Return the image in a streaming response.
    return StreamingResponse(image_bytes, media_type="image/jpeg")
