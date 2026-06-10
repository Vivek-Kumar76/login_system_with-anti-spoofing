from fastapi import FastAPI, File,Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import cv2
import io

from tinyliveness import create_default_onnx_detector

face_app = FaceAnalysis(name= "buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

embeddings_file = "embeddings.pkl"


def load_embeddings():
    try:
        with open(embeddings_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_embeddings(embeddings):
    with open(embeddings_file, "wb") as f:
        pickle.dump(embeddings, f)

detector = create_default_onnx_detector()

app = FastAPI(title="TinyLiveness API")
app.add_middleware( CORSMiddleware,
                    allow_origins=["*"],
                    allow_methods=["*"],
                    allow_headers=["*"])


@app.get("/")
def root():
    return FileResponse("static/index.html", )

@app.get("/register")
def register():
    return FileResponse("static/register.html", )
@app.get("/dashboard")
def dashboard():
    return FileResponse("static/dashboard.html", )

@app.post("/login")
async def login_face(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="File is empty.")

    try:
        pil_img    = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224), Image.LANCZOS)
        face_array = np.array(pil_img, dtype=np.float32)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)   
    result = detector.predict_image(face_array)
    live_probability= round(float(result.live_probability), 6)
    spoof_probability= round(1 - live_probability, 6)

    if spoof_probability > 0.60:
        return {
            "decision": "spoof",
            "live_probability": live_probability,
            "spoof_probability": spoof_probability,
            "message": "Spoof detected — looks like a photo, screen, or replay."
        }

    faces = face_app.get(cv_img)   
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in image.")

    input_embedding = faces[0].embedding   
    embeddings = load_embeddings()
    if not embeddings:
        raise HTTPException(status_code=404, detail="No registered users found.")
    similarity_scores = {
        name: np.dot(input_embedding, emb) / (np.linalg.norm(input_embedding) * np.linalg.norm(emb))
        for name, emb in embeddings.items()
    }
    best_match  = max(similarity_scores, key=similarity_scores.get)
    best_score  = round(float(similarity_scores[best_match]), 4)

    if best_score > 0.60:
        return {
            "decision": "live",
            "live_probability": live_probability,
            "spoof_probability": spoof_probability,
            "similarity_score": best_score,
            "message": f"Welcome back, {best_match}!"
        }
    else:
        raise HTTPException(
            status_code=401,
            detail=f"Face not recognized. Best similarity: {best_score}"
        )

@app.post("/register")
async def register(name: str= Form(...),
                   username: str= Form(...),
                   email: str= Form(...),
                   password: str= Form(...),
                   file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    
    contents = await file.read()

    np_arr = np.frombuffer(contents, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    faces = face_app.get(img)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in the image.")
    face = faces[0]
    embedding = face.embedding

    embeddings = load_embeddings()
    embeddings[name] = embedding
    save_embeddings(embeddings)

    return {
        "message": "User registered successfully."
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)