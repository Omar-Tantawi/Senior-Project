"""Generate a comprehensive PDF technical report about the AI Attendance System."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, ListFlowable, ListItem, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from datetime import date


def build_report():
    output_path = "AI_Attendance_System_Technical_Report.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=25*mm,
        leftMargin=25*mm,
        topMargin=25*mm,
        bottomMargin=25*mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        leading=34,
        textColor=HexColor("#1a1a2e"),
        spaceAfter=10,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        textColor=HexColor("#555555"),
        alignment=TA_CENTER,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=HexColor("#1a1a2e"),
        spaceBefore=20,
        spaceAfter=10,
        borderWidth=1,
        borderColor=HexColor("#1a1a2e"),
        borderPadding=5,
    ))
    styles.add(ParagraphStyle(
        name="SubSection",
        parent=styles["Heading2"],
        fontSize=14,
        leading=17,
        textColor=HexColor("#16213e"),
        spaceBefore=14,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="SubSubSection",
        parent=styles["Heading3"],
        fontSize=12,
        leading=15,
        textColor=HexColor("#0f3460"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontSize=8,
        leading=11,
        backColor=HexColor("#f4f4f4"),
        borderWidth=0.5,
        borderColor=HexColor("#cccccc"),
        borderPadding=6,
        leftIndent=10,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="BulletText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        leftIndent=20,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="TableHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=white,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    ))

    story = []

    # ===== COVER PAGE =====
    story.append(Spacer(1, 80))
    story.append(Paragraph("AI-Based Attendance System", styles["CoverTitle"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Using Face Detection and Recognition", styles["CoverTitle"]))
    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor("#1a1a2e")))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Technical Report", styles["CoverSubtitle"]))
    story.append(Paragraph("Senior Project - School Management System", styles["CoverSubtitle"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Date: {date.today().strftime('%B %d, %Y')}", styles["CoverSubtitle"]))
    story.append(Spacer(1, 40))

    # Tech stack summary table on cover
    tech_data = [
        [Paragraph("<b>Component</b>", styles["TableHeader"]),
         Paragraph("<b>Technology</b>", styles["TableHeader"])],
        [Paragraph("Face Detection", styles["TableCell"]),
         Paragraph("RetinaFace (det_10g.onnx)", styles["TableCell"])],
        [Paragraph("Face Recognition", styles["TableCell"]),
         Paragraph("ArcFace w600k_r50 (512-dim embeddings)", styles["TableCell"])],
        [Paragraph("Inference Engine", styles["TableCell"]),
         Paragraph("ONNX Runtime with CUDA GPU acceleration", styles["TableCell"])],
        [Paragraph("Language", styles["TableCell"]),
         Paragraph("Python 3.10", styles["TableCell"])],
        [Paragraph("Backend Integration", styles["TableCell"]),
         Paragraph("REST API (PostgreSQL)", styles["TableCell"])],
    ]
    tech_table = Table(tech_data, colWidths=[140, 300])
    tech_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(tech_table)
    story.append(PageBreak())

    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("Table of Contents", styles["SectionTitle"]))
    story.append(Spacer(1, 10))
    toc_items = [
        "1. Introduction and Problem Statement",
        "2. System Architecture Overview",
        "3. AI Models - Deep Dive",
        "    3.1. RetinaFace - Face Detection",
        "    3.2. ArcFace - Face Recognition",
        "    3.3. Why These Models Were Chosen",
        "4. Code Architecture",
        "    4.1. Project Structure",
        "    4.2. Core Modules",
        "    4.3. Supporting Modules",
        "5. Processing Pipeline",
        "    5.1. Enrollment Pipeline",
        "    5.2. Attendance Pipeline",
        "6. Data Flow and Backend Integration",
        "7. Configuration and Tuning",
        "8. Hardware Requirements",
        "9. Limitations and Future Work",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["BodyText2"]))
    story.append(PageBreak())

    # ===== 1. INTRODUCTION =====
    story.append(Paragraph("1. Introduction and Problem Statement", styles["SectionTitle"]))
    story.append(Paragraph(
        "This document describes the AI-based attendance system developed as part of a school management "
        "system senior project. The system uses a fixed classroom camera to automatically detect and "
        "recognize student faces, recording attendance without manual intervention.",
        styles["BodyText2"]
    ))
    story.append(Paragraph("<b>Problem Constraints:</b>", styles["BodyText2"]))

    constraints_data = [
        [Paragraph("<b>Constraint</b>", styles["TableHeader"]),
         Paragraph("<b>Detail</b>", styles["TableHeader"]),
         Paragraph("<b>Impact</b>", styles["TableHeader"])],
        [Paragraph("Camera Resolution", styles["TableCell"]),
         Paragraph("720p (worst case)", styles["TableCell"]),
         Paragraph("Faces at distance may be only 30-50 pixels", styles["TableCell"])],
        [Paragraph("Distance", styles["TableCell"]),
         Paragraph("Up to 7 meters", styles["TableCell"]),
         Paragraph("Requires robust small-face detection", styles["TableCell"])],
        [Paragraph("Camera Angle", styles["TableCell"]),
         Paragraph("Top corner or above whiteboard", styles["TableCell"]),
         Paragraph("Non-frontal face angles", styles["TableCell"])],
        [Paragraph("Lighting", styles["TableCell"]),
         Paragraph("Dim room, inconsistent shadows", styles["TableCell"]),
         Paragraph("Need lighting-robust recognition", styles["TableCell"])],
        [Paragraph("Students per Frame", styles["TableCell"]),
         Paragraph("30-50 simultaneously", styles["TableCell"]),
         Paragraph("Multi-face detection + matching at scale", styles["TableCell"])],
        [Paragraph("Attendance Rule", styles["TableCell"]),
         Paragraph("Detected any time = present", styles["TableCell"]),
         Paragraph("No need for continuous tracking", styles["TableCell"])],
    ]
    constraints_table = Table(constraints_data, colWidths=[110, 140, 190])
    constraints_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(constraints_table)
    story.append(PageBreak())

    # ===== 2. SYSTEM ARCHITECTURE =====
    story.append(Paragraph("2. System Architecture Overview", styles["SectionTitle"]))
    story.append(Paragraph(
        "The system follows a modular architecture with clear separation between face detection, "
        "face recognition, attendance logic, and external integration. All AI inference runs locally "
        "using ONNX Runtime with GPU acceleration.",
        styles["BodyText2"]
    ))

    story.append(Paragraph(
        "The architecture consists of the following layers:",
        styles["BodyText2"]
    ))

    arch_text = """<b>Input Layer:</b> USB webcam or IP camera captures 720p video frames at 30 FPS.<br/><br/>
<b>Detection Layer:</b> RetinaFace processes each frame (every 10 seconds by default) and outputs bounding boxes, confidence scores, and 5-point facial landmarks for every face detected.<br/><br/>
<b>Alignment Layer:</b> Each detected face is aligned to a standard 112x112 template using a similarity transform computed from the 5 facial landmarks. This normalizes for rotation, scale, and translation.<br/><br/>
<b>Recognition Layer:</b> The aligned face is passed through ArcFace, which outputs a 512-dimensional embedding vector. This vector is compared against all enrolled student embeddings using cosine similarity.<br/><br/>
<b>Decision Layer:</b> The AttendanceManager records which students have been detected. A student seen at any point during the session is marked "present." At session end, anyone not detected is marked "absent."<br/><br/>
<b>Output Layer:</b> Results are saved to CSV and optionally sent to the backend via REST API."""
    story.append(Paragraph(arch_text, styles["BodyText2"]))

    # Architecture diagram as text
    story.append(Spacer(1, 10))
    arch_diagram = """Camera Frame --> RetinaFace Detection --> Face Alignment (5-point landmarks)
    --> ArcFace Embedding (512-dim) --> Cosine Similarity Match --> Attendance Record
    --> CSV File + REST API (Backend)"""
    story.append(Paragraph(f"<font face='Courier' size=8>{arch_diagram}</font>", styles["BodyText2"]))
    story.append(PageBreak())

    # ===== 3. AI MODELS =====
    story.append(Paragraph("3. AI Models - Deep Dive", styles["SectionTitle"]))

    # 3.1 RetinaFace
    story.append(Paragraph("3.1. RetinaFace - Face Detection", styles["SubSection"]))
    story.append(Paragraph(
        "RetinaFace is a single-stage face detection model that jointly predicts face bounding boxes, "
        "confidence scores, and 5-point facial landmarks (left eye, right eye, nose tip, left mouth corner, "
        "right mouth corner) in a single forward pass.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>Model File:</b> det_10g.onnx (~16 MB)", styles["BodyText2"]))
    story.append(Paragraph("<b>Input:</b> 640x640 RGB image, normalized to [-1, 1]", styles["BodyText2"]))
    story.append(Paragraph("<b>Output:</b> For each of 3 feature map strides (8, 16, 32):", styles["BodyText2"]))

    story.append(Paragraph("- <b>Scores tensor:</b> Detection confidence for each anchor", styles["BulletText"]))
    story.append(Paragraph("- <b>Bounding box deltas:</b> 4 values (x1, y1, x2, y2) relative to anchor", styles["BulletText"]))
    story.append(Paragraph("- <b>Landmark deltas:</b> 10 values (5 keypoints x 2 coordinates)", styles["BulletText"]))

    story.append(Paragraph("<b>How It Works (Technical Detail):</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "RetinaFace uses a Feature Pyramid Network (FPN) backbone that generates feature maps at three "
        "different scales (strides 8, 16, and 32). Stride 8 captures small faces (critical for our 7m "
        "distance scenario where faces may be only 30-50 pixels). Stride 32 captures large, nearby faces. "
        "Each grid cell at each stride has 2 anchors, giving the model dense coverage across all face sizes.",
        styles["BodyText2"]
    ))
    story.append(Paragraph(
        "<b>Post-processing:</b> After inference, the raw model outputs go through: (1) score filtering "
        "(discard detections below the configured threshold, default 0.5), (2) conversion from anchor-relative "
        "deltas to absolute pixel coordinates, and (3) Non-Maximum Suppression (NMS) with IoU threshold 0.4 "
        "to remove duplicate detections of the same face.",
        styles["BodyText2"]
    ))
    story.append(Paragraph(
        "<b>Why RetinaFace over YOLO for faces?</b> At 720p with faces at 7m, many faces will be very small. "
        "RetinaFace's multi-scale FPN with stride-8 anchors is specifically designed for small face detection. "
        "Standard YOLO models are general object detectors that may miss tiny faces. RetinaFace also outputs "
        "facial landmarks directly, which are essential for face alignment before recognition.",
        styles["BodyText2"]
    ))

    # 3.2 ArcFace
    story.append(Paragraph("3.2. ArcFace - Face Recognition", styles["SubSection"]))
    story.append(Paragraph(
        "ArcFace (Additive Angular Margin Loss) is a face recognition model that converts a face image into "
        "a compact 512-dimensional embedding vector. Two faces of the same person will produce embedding vectors "
        "with high cosine similarity, while different people will produce dissimilar vectors.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>Model File:</b> w600k_r50.onnx (~167 MB)", styles["BodyText2"]))
    story.append(Paragraph("<b>Architecture:</b> ResNet-50 backbone", styles["BodyText2"]))
    story.append(Paragraph("<b>Training Data:</b> 600,000 identities (w600k)", styles["BodyText2"]))
    story.append(Paragraph("<b>Input:</b> 112x112 aligned face image, normalized to [-1, 1]", styles["BodyText2"]))
    story.append(Paragraph("<b>Output:</b> 512-dimensional L2-normalized embedding vector", styles["BodyText2"]))

    story.append(Paragraph("<b>Face Alignment (Pre-processing):</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "Before feeding a face to ArcFace, it must be aligned to a standard template. This is done using "
        "the 5 facial landmarks detected by RetinaFace. A similarity transform (rotation + scale + translation) "
        "is computed to map the detected landmarks to a fixed reference template. The face is then warped to "
        "produce a normalized 112x112 image. This alignment ensures that the eyes, nose, and mouth are always "
        "in consistent positions, dramatically improving recognition accuracy regardless of head pose.",
        styles["BodyText2"]
    ))

    story.append(Paragraph(
        "The standard ArcFace reference points for 112x112 are: Left Eye (38.29, 51.70), "
        "Right Eye (73.53, 51.50), Nose (56.03, 71.74), Left Mouth (41.55, 92.37), Right Mouth (70.73, 92.20).",
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>How ArcFace Loss Works:</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "During training, ArcFace adds an angular margin penalty to the softmax loss function. This forces "
        "the model to learn embeddings where same-person vectors are tightly clustered and different-person "
        "vectors are pushed far apart on a hypersphere. The result is highly discriminative embeddings that "
        "achieve 99.40% accuracy on the LFW benchmark and 98.36% on MegaFace.",
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>Matching Process:</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "At runtime, cosine similarity is computed between the query embedding and all enrolled embeddings. "
        "The enrolled student with the highest similarity above the threshold (default 0.4) is returned as "
        "the match. If no match exceeds the threshold, the face is marked as 'unknown.'",
        styles["BodyText2"]
    ))

    # 3.3 Why these models
    story.append(Paragraph("3.3. Why These Models Were Chosen", styles["SubSection"]))

    comparison_data = [
        [Paragraph("<b>Criterion</b>", styles["TableHeader"]),
         Paragraph("<b>RetinaFace + ArcFace</b>", styles["TableHeader"]),
         Paragraph("<b>YOLO + FaceNet</b>", styles["TableHeader"]),
         Paragraph("<b>MTCNN + FaceNet</b>", styles["TableHeader"])],
        [Paragraph("Small face detection", styles["TableCell"]),
         Paragraph("Excellent (multi-scale FPN)", styles["TableCell"]),
         Paragraph("Good", styles["TableCell"]),
         Paragraph("Poor at very small sizes", styles["TableCell"])],
        [Paragraph("Lighting robustness", styles["TableCell"]),
         Paragraph("Excellent (ArcFace)", styles["TableCell"]),
         Paragraph("Good", styles["TableCell"]),
         Paragraph("Good", styles["TableCell"])],
        [Paragraph("Angle tolerance", styles["TableCell"]),
         Paragraph("Good (landmark alignment)", styles["TableCell"]),
         Paragraph("Moderate", styles["TableCell"]),
         Paragraph("Moderate", styles["TableCell"])],
        [Paragraph("Speed (50 faces)", styles["TableCell"]),
         Paragraph("~50ms on GPU", styles["TableCell"]),
         Paragraph("~28ms detection only", styles["TableCell"]),
         Paragraph("~200ms", styles["TableCell"])],
        [Paragraph("LFW Accuracy", styles["TableCell"]),
         Paragraph("99.40%", styles["TableCell"]),
         Paragraph("99.65% (FaceNet512)", styles["TableCell"]),
         Paragraph("99.65%", styles["TableCell"])],
        [Paragraph("Training data scale", styles["TableCell"]),
         Paragraph("600K identities", styles["TableCell"]),
         Paragraph("Variable", styles["TableCell"]),
         Paragraph("Variable", styles["TableCell"])],
    ]
    comparison_table = Table(comparison_data, colWidths=[100, 120, 110, 110])
    comparison_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("BACKGROUND", (1, 1), (1, -1), HexColor("#e8f5e9")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(comparison_table)
    story.append(Paragraph(
        "<i>RetinaFace + ArcFace (highlighted) was chosen because small face detection and lighting robustness "
        "are the highest priorities for our classroom scenario.</i>",
        styles["BodyText2"]
    ))
    story.append(PageBreak())

    # ===== 4. CODE ARCHITECTURE =====
    story.append(Paragraph("4. Code Architecture", styles["SectionTitle"]))

    # 4.1 Project Structure
    story.append(Paragraph("4.1. Project Structure", styles["SubSection"]))
    struct = """attendance_system/
+-- config.yaml                  # All tunable parameters
+-- requirements.txt             # Python dependencies
+-- main.py                      # Entry point: live attendance
+-- enroll.py                    # Entry point: student enrollment
+-- download_models.py           # Downloads ONNX model files
+-- models/
|   +-- det_10g.onnx             # RetinaFace detection (~16MB)
|   +-- w600k_r50.onnx           # ArcFace recognition (~167MB)
+-- core/
|   +-- detector.py              # Face detection + embedding extraction
|   +-- recognizer.py            # Embedding database + matching
|   +-- attendance_manager.py    # Attendance tracking logic
+-- api/
|   +-- client.py                # REST API client for backend
+-- storage/
|   +-- csv_handler.py           # CSV report writer
+-- utils/
|   +-- visualization.py         # Bounding box + label drawing
+-- data/
    +-- embeddings/              # Enrolled student face data
    +-- unknown_faces/           # Captured unknown face images
    +-- attendance_csv/          # Generated CSV reports"""
    story.append(Paragraph(f"<font face='Courier' size=8>{struct}</font>", styles["BodyText2"]))

    # 4.2 Core Modules
    story.append(Paragraph("4.2. Core Modules", styles["SubSection"]))

    # detector.py
    story.append(Paragraph("<b>core/detector.py - FaceDetector Class</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "This is the main AI module. It loads two ONNX models (RetinaFace + ArcFace) using ONNX Runtime "
        "and provides a single detect_faces() method that takes a BGR frame and returns a list of Face objects.",
        styles["BodyText2"]
    ))

    detector_details = [
        [Paragraph("<b>Method</b>", styles["TableHeader"]),
         Paragraph("<b>Purpose</b>", styles["TableHeader"]),
         Paragraph("<b>Details</b>", styles["TableHeader"])],
        [Paragraph("__init__()", styles["TableCell"]),
         Paragraph("Load ONNX models", styles["TableCell"]),
         Paragraph("Loads det_10g.onnx and w600k_r50.onnx into ONNX Runtime sessions. Prefers CUDA, falls back to CPU.", styles["TableCell"])],
        [Paragraph("_preprocess_det()", styles["TableCell"]),
         Paragraph("Prepare frame for detection", styles["TableCell"]),
         Paragraph("Resizes to 640x640, pads, normalizes pixels to [-1,1], transposes to NCHW format.", styles["TableCell"])],
        [Paragraph("_postprocess_det()", styles["TableCell"]),
         Paragraph("Parse model outputs", styles["TableCell"]),
         Paragraph("Decodes anchor-based outputs at 3 strides, applies score filtering and NMS.", styles["TableCell"])],
        [Paragraph("_extract_embedding()", styles["TableCell"]),
         Paragraph("Get face identity vector", styles["TableCell"]),
         Paragraph("Aligns face to 112x112, normalizes, runs ArcFace, returns L2-normalized 512-dim vector.", styles["TableCell"])],
        [Paragraph("detect_faces()", styles["TableCell"]),
         Paragraph("Main entry point", styles["TableCell"]),
         Paragraph("Runs detection + embedding extraction. Returns list of Face dataclass objects.", styles["TableCell"])],
    ]
    det_table = Table(detector_details, colWidths=[110, 100, 230])
    det_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(det_table)

    story.append(Paragraph(
        "<b>Key helper functions:</b> _distance2bbox() converts anchor deltas to pixel coordinates. "
        "_nms() removes overlapping duplicate detections. _align_face() warps detected faces to the "
        "standard ArcFace template using cv2.estimateAffinePartial2D().",
        styles["BodyText2"]
    ))

    # recognizer.py
    story.append(Paragraph("<b>core/recognizer.py - FaceRecognizer Class</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "Manages the enrolled student database. Each student's face embeddings are stored as .npy files "
        "on disk (one directory per student). At startup, all embeddings are loaded into memory for fast matching.",
        styles["BodyText2"]
    ))
    story.append(Paragraph(
        "<b>Storage format:</b> data/embeddings/{student_id}/embeddings.npy (Nx512 array) + name.txt",
        styles["BodyText2"]
    ))
    story.append(Paragraph(
        "<b>Matching algorithm:</b> For each query face, cosine similarity is computed against every enrolled "
        "student's embeddings (using numpy dot product and norm). The student with the highest similarity above "
        "the threshold (0.4 by default) is returned. With 50 students and 5 embeddings each, this is 250 "
        "similarity computations per face - trivial for modern hardware.",
        styles["BodyText2"]
    ))

    # attendance_manager.py
    story.append(Paragraph("<b>core/attendance_manager.py - AttendanceManager Class</b>", styles["SubSubSection"]))
    story.append(Paragraph(
        "Implements the attendance business logic. Maintains an in-memory dictionary of detected students. "
        "Each entry tracks: first_seen time, last_seen time, best confidence score, and total detection count. "
        "A student detected at any point during the session is marked 'present.' At session end, "
        "get_full_report() compares detected students against the expected roster to determine who is absent.",
        styles["BodyText2"]
    ))
    story.append(PageBreak())

    # 4.3 Supporting Modules
    story.append(Paragraph("4.3. Supporting Modules", styles["SubSection"]))

    support_data = [
        [Paragraph("<b>Module</b>", styles["TableHeader"]),
         Paragraph("<b>File</b>", styles["TableHeader"]),
         Paragraph("<b>Responsibility</b>", styles["TableHeader"])],
        [Paragraph("API Client", styles["TableCell"]),
         Paragraph("api/client.py", styles["TableCell"]),
         Paragraph("Communicates with the school management backend via REST. Fetches student roster and schedule, submits attendance records. Gracefully handles backend unavailability (offline mode).", styles["TableCell"])],
        [Paragraph("CSV Handler", styles["TableCell"]),
         Paragraph("storage/csv_handler.py", styles["TableCell"]),
         Paragraph("Writes attendance reports to CSV files with columns: student_id, student_name, status, first_seen, last_seen, best_confidence, detection_count.", styles["TableCell"])],
        [Paragraph("Visualization", styles["TableCell"]),
         Paragraph("utils/visualization.py", styles["TableCell"]),
         Paragraph("Draws bounding boxes (green=recognized, red=unknown), name labels with confidence scores, and a status bar showing present count and FPS.", styles["TableCell"])],
    ]
    support_table = Table(support_data, colWidths=[80, 100, 260])
    support_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(support_table)
    story.append(PageBreak())

    # ===== 5. PROCESSING PIPELINE =====
    story.append(Paragraph("5. Processing Pipeline", styles["SectionTitle"]))

    # 5.1 Enrollment
    story.append(Paragraph("5.1. Enrollment Pipeline", styles["SubSection"]))
    story.append(Paragraph("<b>Command:</b>", styles["BodyText2"]))
    story.append(Paragraph(
        '<font face="Courier" size=9>python enroll.py --student-id 12345 --student-name "John Doe"</font>',
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>Step-by-step process:</b>", styles["BodyText2"]))
    enroll_steps = [
        "1. Open webcam and load RetinaFace + ArcFace models",
        "2. Display live camera feed with face detection overlays",
        "3. Prompt user for 5 poses: FRONT, SLIGHT LEFT, SLIGHT RIGHT, FRONT (natural), FRONT (different expression)",
        "4. On SPACE press: detect all faces, select the largest (closest to camera), extract 512-dim embedding",
        "5. After 5 captures (minimum 3): stack embeddings into (5, 512) numpy array",
        "6. Save to data/embeddings/{student_id}/embeddings.npy + name.txt",
        "7. Student is now enrolled and can be recognized in attendance mode",
    ]
    for step in enroll_steps:
        story.append(Paragraph(step, styles["BulletText"]))

    story.append(Paragraph(
        "<b>Why 5 photos with different poses?</b> Multiple embeddings from different angles create a more "
        "robust representation. During matching, we compute similarity against ALL enrolled embeddings and "
        "take the maximum. This means a student can be recognized even if their current head angle doesn't "
        "exactly match the enrollment front pose.",
        styles["BodyText2"]
    ))

    # 5.2 Attendance
    story.append(Paragraph("5.2. Attendance Pipeline", styles["SubSection"]))
    story.append(Paragraph("<b>Command:</b>", styles["BodyText2"]))
    story.append(Paragraph(
        '<font face="Courier" size=9>python main.py --duration 60 --interval 10 --class-id "CS101"</font>',
        styles["BodyText2"]
    ))

    story.append(Paragraph("<b>Step-by-step process:</b>", styles["BodyText2"]))
    attend_steps = [
        "1. Load models, enrolled student database, and config",
        "2. Try connecting to backend API; if unavailable, run in offline mode (CSV only)",
        "3. If API available, fetch expected student roster for this class",
        "4. Open camera and start main loop",
        "5. Every 10 seconds (configurable): run face detection on current frame",
        "6. For each detected face: align to 112x112, extract ArcFace embedding",
        "7. Compare embedding against all enrolled students (cosine similarity)",
        "8. If similarity >= 0.4: mark student as PRESENT in AttendanceManager",
        "9. If similarity < 0.3: save cropped face image as 'unknown' for review",
        "10. Display live video with bounding boxes, names, confidence, and status bar",
        "11. After duration expires (or Q pressed): generate attendance report",
        "12. Save report to CSV + submit to backend API",
        "13. Print summary: X present, Y absent, Z unknown",
    ]
    for step in attend_steps:
        story.append(Paragraph(step, styles["BulletText"]))
    story.append(PageBreak())

    # ===== 6. DATA FLOW =====
    story.append(Paragraph("6. Data Flow and Backend Integration", styles["SectionTitle"]))

    story.append(Paragraph("<b>What the AI receives from the backend:</b>", styles["SubSubSection"]))
    recv_data = [
        [Paragraph("<b>Endpoint</b>", styles["TableHeader"]),
         Paragraph("<b>Method</b>", styles["TableHeader"]),
         Paragraph("<b>Data</b>", styles["TableHeader"]),
         Paragraph("<b>When</b>", styles["TableHeader"])],
        [Paragraph("GET /students", styles["TableCell"]),
         Paragraph("GET", styles["TableCell"]),
         Paragraph("student_id, student_name list", styles["TableCell"]),
         Paragraph("On startup", styles["TableCell"])],
        [Paragraph("GET /schedule", styles["TableCell"]),
         Paragraph("GET", styles["TableCell"]),
         Paragraph("class_id, room_id, start/end time, student list", styles["TableCell"]),
         Paragraph("On startup", styles["TableCell"])],
    ]
    recv_table = Table(recv_data, colWidths=[100, 50, 170, 120])
    recv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(recv_table)

    story.append(Paragraph("<b>What the AI sends to the backend:</b>", styles["SubSubSection"]))
    send_data = [
        [Paragraph("<b>Endpoint</b>", styles["TableHeader"]),
         Paragraph("<b>Method</b>", styles["TableHeader"]),
         Paragraph("<b>Data</b>", styles["TableHeader"]),
         Paragraph("<b>When</b>", styles["TableHeader"])],
        [Paragraph("POST /attendance", styles["TableCell"]),
         Paragraph("POST", styles["TableCell"]),
         Paragraph("date, class_id, present list (id, name, confidence, timestamps), absent list", styles["TableCell"]),
         Paragraph("At session end", styles["TableCell"])],
        [Paragraph("POST /enroll", styles["TableCell"]),
         Paragraph("POST", styles["TableCell"]),
         Paragraph("student_id, student_name", styles["TableCell"]),
         Paragraph("After enrollment", styles["TableCell"])],
    ]
    send_table = Table(send_data, colWidths=[100, 50, 170, 120])
    send_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(send_table)

    story.append(Paragraph(
        "<b>Offline mode:</b> If the backend is unavailable, the system runs fully standalone. "
        "Attendance is saved only to CSV. The API client gracefully handles connection failures "
        "and prints a warning at startup.",
        styles["BodyText2"]
    ))
    story.append(PageBreak())

    # ===== 7. CONFIGURATION =====
    story.append(Paragraph("7. Configuration and Tuning", styles["SectionTitle"]))
    story.append(Paragraph(
        "All parameters are centralized in config.yaml. No code changes are needed to tune the system.",
        styles["BodyText2"]
    ))

    config_data = [
        [Paragraph("<b>Parameter</b>", styles["TableHeader"]),
         Paragraph("<b>Default</b>", styles["TableHeader"]),
         Paragraph("<b>Purpose</b>", styles["TableHeader"]),
         Paragraph("<b>Tuning Advice</b>", styles["TableHeader"])],
        [Paragraph("det_thresh", styles["TableCell"]),
         Paragraph("0.5", styles["TableCell"]),
         Paragraph("Min face detection confidence", styles["TableCell"]),
         Paragraph("Lower to detect more distant faces; raise to reduce false positives", styles["TableCell"])],
        [Paragraph("det_size", styles["TableCell"]),
         Paragraph("[640, 640]", styles["TableCell"]),
         Paragraph("Detection model input resolution", styles["TableCell"]),
         Paragraph("Higher = better small face detection but slower", styles["TableCell"])],
        [Paragraph("similarity_threshold", styles["TableCell"]),
         Paragraph("0.4", styles["TableCell"]),
         Paragraph("Min cosine similarity for a match", styles["TableCell"]),
         Paragraph("Lower = more matches but more false IDs; higher = stricter but may miss", styles["TableCell"])],
        [Paragraph("unknown_threshold", styles["TableCell"]),
         Paragraph("0.3", styles["TableCell"]),
         Paragraph("Below this, face is 'unknown'", styles["TableCell"]),
         Paragraph("Gap between this and similarity_threshold is the 'uncertain' zone", styles["TableCell"])],
        [Paragraph("capture_interval_sec", styles["TableCell"]),
         Paragraph("10", styles["TableCell"]),
         Paragraph("Seconds between face captures", styles["TableCell"]),
         Paragraph("Lower = more captures, better attendance, but higher GPU load", styles["TableCell"])],
        [Paragraph("session_duration_min", styles["TableCell"]),
         Paragraph("60", styles["TableCell"]),
         Paragraph("How long to run attendance", styles["TableCell"]),
         Paragraph("Set to match your attendance period length", styles["TableCell"])],
        [Paragraph("max_faces", styles["TableCell"]),
         Paragraph("60", styles["TableCell"]),
         Paragraph("Max faces to process per frame", styles["TableCell"]),
         Paragraph("Set to slightly above your largest class size", styles["TableCell"])],
        [Paragraph("photos_per_student", styles["TableCell"]),
         Paragraph("5", styles["TableCell"]),
         Paragraph("Enrollment photos count", styles["TableCell"]),
         Paragraph("More photos = better recognition across angles", styles["TableCell"])],
    ]
    config_table = Table(config_data, colWidths=[90, 55, 130, 165])
    config_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(config_table)
    story.append(PageBreak())

    # ===== 8. HARDWARE =====
    story.append(Paragraph("8. Hardware Requirements", styles["SectionTitle"]))

    hw_data = [
        [Paragraph("<b>Component</b>", styles["TableHeader"]),
         Paragraph("<b>Minimum</b>", styles["TableHeader"]),
         Paragraph("<b>Recommended</b>", styles["TableHeader"])],
        [Paragraph("GPU", styles["TableCell"]),
         Paragraph("NVIDIA RTX 3050 (8GB)", styles["TableCell"]),
         Paragraph("NVIDIA RTX 5060 (8GB) or better", styles["TableCell"])],
        [Paragraph("CPU", styles["TableCell"]),
         Paragraph("Any modern quad-core", styles["TableCell"]),
         Paragraph("Intel i5/AMD Ryzen 5 or better", styles["TableCell"])],
        [Paragraph("RAM", styles["TableCell"]),
         Paragraph("8 GB", styles["TableCell"]),
         Paragraph("16 GB", styles["TableCell"])],
        [Paragraph("Storage", styles["TableCell"]),
         Paragraph("500 MB (models + code)", styles["TableCell"]),
         Paragraph("1 GB+ (for unknown face images)", styles["TableCell"])],
        [Paragraph("Camera", styles["TableCell"]),
         Paragraph("720p USB webcam", styles["TableCell"]),
         Paragraph("1080p IP camera", styles["TableCell"])],
        [Paragraph("Python", styles["TableCell"]),
         Paragraph("3.10", styles["TableCell"]),
         Paragraph("3.10", styles["TableCell"])],
        [Paragraph("CUDA", styles["TableCell"]),
         Paragraph("Bundled with onnxruntime-gpu", styles["TableCell"]),
         Paragraph("CUDA 12.x + cuDNN (optional)", styles["TableCell"])],
    ]
    hw_table = Table(hw_data, colWidths=[80, 175, 185])
    hw_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(hw_table)

    story.append(Paragraph(
        "<b>Performance estimate:</b> On an RTX 3050, processing 50 faces per frame takes approximately "
        "50-80ms (detection ~30ms + 50x recognition ~1ms each). With a 10-second capture interval, "
        "GPU utilization stays below 5%, leaving ample headroom for other tasks.",
        styles["BodyText2"]
    ))
    story.append(PageBreak())

    # ===== 9. LIMITATIONS AND FUTURE =====
    story.append(Paragraph("9. Limitations and Future Work", styles["SectionTitle"]))

    story.append(Paragraph("<b>Current Limitations:</b>", styles["SubSection"]))
    limitations = [
        "Face masks significantly reduce recognition accuracy (covering nose and mouth removes key landmarks and facial features).",
        "At 7m distance on 720p, some faces may be too small for reliable recognition even with RetinaFace's multi-scale detection.",
        "The system uses pretrained models. Fine-tuning on actual student data would improve accuracy for the specific population.",
        "Cosine similarity matching is linear (O(n) per face). For very large databases (1000+ students), approximate nearest neighbor search would be needed.",
        "No liveness detection - the system cannot distinguish a real face from a photograph (spoofing vulnerability).",
        "Recognition accuracy degrades with extreme head poses (looking straight down or away from camera).",
    ]
    for lim in limitations:
        story.append(Paragraph(f"- {lim}", styles["BulletText"]))

    story.append(Paragraph("<b>Planned Future Improvements:</b>", styles["SubSection"]))
    future = [
        "Fine-tune ArcFace on the actual student population to improve accuracy on local/cultural features.",
        "Add liveness detection to prevent spoofing with photos.",
        "Implement face tracking between frames to avoid re-detecting the same person repeatedly.",
        "Upgrade to 1080p cameras for better small-face detection at distance.",
        "Add anti-spoofing measures (3D depth, texture analysis).",
        "Integrate with fight detection model for behavioral monitoring.",
        "Support multiple cameras per classroom for better coverage.",
        "Implement incremental enrollment (add photos over time to improve each student's embedding database).",
    ]
    for item in future:
        story.append(Paragraph(f"- {item}", styles["BulletText"]))

    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc")))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<i>Report generated on {date.today().strftime('%B %d, %Y')}. "
        "This system is part of the School Management System senior project.</i>",
        styles["BodyText2"]
    ))

    # Build PDF
    doc.build(story)
    print(f"\nPDF report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
