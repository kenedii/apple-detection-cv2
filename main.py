import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from apple_detection import detect_apples

app = FastAPI(title="Apple Detector API")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Apple Detector API! Visit /docs for the API documentation."}

@app.post("/is_apple")
async def is_apple(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"): # Ensure the uploaded file is an image
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read() # Read the file into memory.
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    
    if image is None: # Check that the image was correctly decoded
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    processed_image, detected_apples = detect_apples(image) # Detect apples in the image
    # detected_apples is a list of bounding boxes/circles of the detected apples

    # Encode the processed image to JPEG format
    success, encoded_image = cv2.imencode(".jpg", processed_image)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    # Convert the encoded image to bytes
    image_bytes = io.BytesIO(encoded_image.tobytes())

    # Return the image in a streaming response
    return StreamingResponse(image_bytes, media_type="image/jpeg")
