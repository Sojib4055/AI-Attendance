# Iris-Based Attendance System (Skeleton with Working Pipeline)

This project is a **full code skeleton** for an iris-based attendance system
using the conceptual pipeline:

> IriTrack → RITnet → Daugman Normalization → DeepIrisNet2 → Cosine Similarity

Because the actual published research models (IriTrack, RITnet, DeepIrisNet2)
and their weights are not shipped here, this repo includes **working
placeholders**:

- `IrisDetector` uses OpenCV's Haar `haarcascade_eye.xml` to find eyes.
- `IrisSegmenter` approximates the iris region using a Hough circle transform.
- `DaugmanNormalizer` implements a real rubber-sheet normalization.
- `IrisEncoder` is a lightweight CNN that stands in for DeepIrisNet2.
  You can later replace it with the real architecture & weights.
- `IrisMatcher` uses cosine similarity on embeddings.

This codebase is meant to be a **starting point**:
- You can run it end-to-end with test videos.
- Then progressively replace the placeholder models with real biometric-grade
  models as you obtain them.

## Quick Start

1. Create a virtualenv and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Prepare the placeholder model assets (downloads the Haar eye cascade and creates the model folders):

   ```bash
   python scripts/setup_assets.py
   ```

   > The pipeline will also attempt to auto-download the cascade the first time it runs, but running the script up-front avoids needing internet access at runtime.

3. Enroll a person:

   ```bash
   python scripts/enroll_person.py \
     --name "Alice" \
     --employee_code ALI001 \
     --video data/enrollment_alice.mp4
   ```

4. Process a recorded CCTV video for attendance:

   ```bash
   python scripts/process_video.py \
     --video data/cctv_sample.mp4 \
     --camera_id CAM01
   ```

5. Inspect the SQLite DB `iris_attendance.db` to see:
   - `persons`
   - `iris_templates`
   - `attendance_events`

## Streamlit UI

If you prefer a UI instead of the CLI scripts, launch the Streamlit control panel:

```bash
streamlit run ui/app.py
```

The UI exposes three tabs:

- **Enroll** – fill in the person's details, upload an enrollment clip, and run the pipeline end-to-end.
- **Attendance** – upload a CCTV/monitoring clip plus a camera ID to log matches as attendance events.
- **Database** – inspect enrollment counts and the latest attendance activity coming from `iris_attendance.db`.

## FastAPI Service

For programmatic control or to integrate with other systems, fire up the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

Key endpoints:
- `POST /enroll` (multipart form) — fields `name`, `employee_code`, `department` plus `video` upload.
- `POST /attendance/process` (multipart form) — fields `camera_id` plus `video` upload.
- `GET /persons` — list enrolled personnel with template counts.
- `GET /attendance/events?limit=50` — latest attendance events (limit clamped to 200).
- `GET /health` — lightweight status probe.

Use any HTTP client (curl, Thunder Client, Postman, etc.) to hit these endpoints once the server is running.

## IMPORTANT

- This is **NOT** production biometric accuracy yet.
- To reach your target (IriTrack + RITnet + DeepIrisNet2 quality) you must:
  - Replace `core/iris_detector.py` with a real IriTrack implementation.
  - Replace `core/iris_segmenter.py` with a trained RITnet model.
  - Replace `core/encoder.py`'s `SimpleIrisEncoderNet` with DeepIrisNet2 and load its weights.

The rest of the system (DB, services, pipeline orchestration) is ready to
support a real iris-based attendance deployment.
