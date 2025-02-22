import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from apple_detection import detect_apples, draw_circles
from banana_detection import detect_bananas, draw_boxes

app = FastAPI(title="Apple Detector API")

@app.get("/")
async def read_root():
    return {"message": "Visit /docs for the API documentation."}

@app.post("/is_apple")
async def is_apple(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read() # Read the file into memory.
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR_BGR)

    
    if image is None: # Ensure the image was correctly decoded
        raise HTTPException(status_code=400, detail="Could not decode the image.")

    processed_image, detected_apples = detect_apples(image) # Detect apples in the image
    processed_image, detected_bananas = detect_bananas(processed_image) # Detect bananas in the image
    processed_image = draw_circles(processed_image, detected_apples) # Draw circles around detected apples
    processed_image = draw_boxes(processed_image, detected_bananas) # Draw bounding boxes around detected bananas

    detections = {
        "apples": [detected_apples.size, detected_apples],
        "bananas": [detected_bananas.size, detected_bananas]
    }

    # Encode the processed image to JPEG format
    success, encoded_image = cv2.imencode(".jpg", processed_image)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed.")

    # Convert the encoded image to bytes
    image_bytes = io.BytesIO(encoded_image.tobytes())

    # Return the image in a streaming response
    return StreamingResponse(image_bytes, media_type="image/jpeg")
