"""Generate a comprehensive deep-dive PDF report about the AI Attendance System."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from datetime import date


def build_report():
    output_path = "AI_Attendance_System_Full_Report.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=22*mm,
        leftMargin=22*mm,
        topMargin=22*mm,
        bottomMargin=22*mm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontSize=26, leading=32,
        textColor=HexColor("#1a1a2e"), spaceAfter=10, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="CoverSub", parent=styles["Normal"], fontSize=13, leading=17,
        textColor=HexColor("#555555"), alignment=TA_CENTER, spaceAfter=6))
    styles.add(ParagraphStyle(name="S1", parent=styles["Heading1"], fontSize=17, leading=21,
        textColor=HexColor("#1a1a2e"), spaceBefore=18, spaceAfter=10))
    styles.add(ParagraphStyle(name="S2", parent=styles["Heading2"], fontSize=13, leading=16,
        textColor=HexColor("#16213e"), spaceBefore=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="S3", parent=styles["Heading3"], fontSize=11, leading=14,
        textColor=HexColor("#0f3460"), spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="B", parent=styles["Normal"], fontSize=10, leading=14,
        alignment=TA_JUSTIFY, spaceAfter=8))
    styles.add(ParagraphStyle(name="Bul", parent=styles["Normal"], fontSize=10, leading=14,
        leftIndent=20, spaceAfter=4))
    styles.add(ParagraphStyle(name="CodeBlock", parent=styles["Code"], fontSize=8, leading=11,
        backColor=HexColor("#f4f4f4"), borderWidth=0.5, borderColor=HexColor("#cccccc"),
        borderPadding=6, leftIndent=10, spaceAfter=8, wordWrap='CJK'))
    styles.add(ParagraphStyle(name="TH", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=white, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="TC", parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Math", parent=styles["Normal"], fontSize=10, leading=15,
        alignment=TA_CENTER, spaceBefore=6, spaceAfter=6, backColor=HexColor("#f9f9f9"),
        borderWidth=0.5, borderColor=HexColor("#dddddd"), borderPadding=8))

    def tbl(data, widths):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f8f8f8")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return t

    story = []

    # ===== COVER PAGE =====
    story.append(Spacer(1, 60))
    story.append(Paragraph("AI-Based Attendance System", styles["CoverTitle"]))
    story.append(Paragraph("Complete Technical Deep-Dive", styles["CoverTitle"]))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="60%", thickness=2, color=HexColor("#1a1a2e")))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Models, Algorithms, Mathematics, Code Architecture,", styles["CoverSub"]))
    story.append(Paragraph("Fine-Tuning Guide, API Reference, and Deployment", styles["CoverSub"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Senior Project - School Management System", styles["CoverSub"]))
    story.append(Paragraph(f"{date.today().strftime('%B %d, %Y')}", styles["CoverSub"]))
    story.append(PageBreak())

    # ===== TABLE OF CONTENTS =====
    story.append(Paragraph("Table of Contents", styles["S1"]))
    toc = [
        "1. System Overview",
        "2. RetinaFace - Face Detection Model",
        "    2.1. Architecture (Feature Pyramid Network)",
        "    2.2. Multi-Scale Anchor System",
        "    2.3. Loss Functions and Training",
        "    2.4. Post-Processing: NMS Algorithm",
        "    2.5. Our Implementation (detector.py)",
        "3. ArcFace - Face Recognition Model",
        "    3.1. ResNet-50 Backbone Architecture",
        "    3.2. ArcFace Loss Function (The Core Math)",
        "    3.3. Embedding Space and Cosine Similarity",
        "    3.4. Face Alignment (Similarity Transform)",
        "    3.5. Our Implementation (recognizer.py)",
        "4. How We Handle Difficult Conditions",
        "    4.1. Poor Lighting / Shadows",
        "    4.2. Angled Camera (Non-Frontal Faces)",
        "    4.3. Small / Distant Faces (7 meters)",
        "    4.4. Many Faces (30-50 students)",
        "5. Fine-Tuning Guide",
        "    5.1. Can We Fine-Tune? What and Why?",
        "    5.2. Datasets for Fine-Tuning",
        "    5.3. Fine-Tuning Strategy",
        "6. Complete Code Walkthrough",
        "    6.1. detector.py - Line by Line",
        "    6.2. recognizer.py - Line by Line",
        "    6.3. attendance_manager.py",
        "    6.4. main.py - The Main Loop",
        "    6.5. enroll.py - Enrollment Flow",
        "7. How to Run the System",
        "8. API Reference (Backend Integration)",
        "    8.1. What You Give the Backend",
        "    8.2. What You Get from the Backend",
        "    8.3. Data Schemas",
        "9. Configuration Reference",
        "10. Limitations and When It May Fail",
    ]
    for item in toc:
        story.append(Paragraph(item, styles["B"]))
    story.append(PageBreak())

    # ===== 1. SYSTEM OVERVIEW =====
    story.append(Paragraph("1. System Overview", styles["S1"]))
    story.append(Paragraph(
        "The system uses two neural networks in sequence: <b>RetinaFace</b> for finding faces in the camera "
        "frame, and <b>ArcFace</b> for identifying whose face it is. Both models run as ONNX files on the GPU "
        "using ONNX Runtime, without needing any special Python packages beyond numpy, opencv, and onnxruntime.",
        styles["B"]))
    story.append(Paragraph(
        "<b>The pipeline for each frame:</b> (1) Camera captures 720p BGR image. (2) Image is resized and "
        "normalized for RetinaFace. (3) RetinaFace outputs bounding boxes + 5 landmarks per face. (4) Each "
        "face is aligned to 112x112 using landmarks. (5) ArcFace converts each aligned face to a 512-dim "
        "vector. (6) Vector is compared to enrolled students using cosine similarity. (7) Best match above "
        "threshold = student identified. (8) AttendanceManager records who was seen.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 2. RETINAFACE =====
    story.append(Paragraph("2. RetinaFace - Face Detection Model", styles["S1"]))
    story.append(Paragraph(
        "RetinaFace (Deng et al., 2020, CVPR) is a single-stage face detector that simultaneously predicts "
        "face locations, confidence scores, 5 facial landmarks, and optionally 3D face shape in one forward pass. "
        "We use the 'det_10g' variant (~16MB ONNX file) which outputs bounding boxes + landmarks.",
        styles["B"]))

    # 2.1 FPN
    story.append(Paragraph("2.1. Architecture: Feature Pyramid Network (FPN)", styles["S2"]))
    story.append(Paragraph(
        "The backbone of RetinaFace is a convolutional neural network (CNN) that processes the input image "
        "through progressively deeper layers. As the image passes through the network, it creates <b>feature maps</b> "
        "at different resolutions - think of these as the network's 'understanding' of the image at different "
        "zoom levels.",
        styles["B"]))
    story.append(Paragraph(
        "The <b>Feature Pyramid Network (FPN)</b> takes these multi-resolution feature maps and connects them "
        "with top-down pathways and lateral connections. This means:",
        styles["B"]))
    story.append(Paragraph("- <b>Deep layers</b> (low resolution) have rich semantic meaning - they understand 'this is a face'", styles["Bul"]))
    story.append(Paragraph("- <b>Shallow layers</b> (high resolution) have precise location information - they know exactly where edges are", styles["Bul"]))
    story.append(Paragraph("- <b>FPN combines both</b>: each level gets both semantic and spatial information", styles["Bul"]))
    story.append(Paragraph(
        "RetinaFace produces feature maps at <b>3 strides: 8, 16, and 32</b>. For a 640x640 input image:",
        styles["B"]))

    stride_data = [
        [Paragraph("<b>Stride</b>", styles["TH"]), Paragraph("<b>Feature Map Size</b>", styles["TH"]),
         Paragraph("<b>Each Cell Covers</b>", styles["TH"]), Paragraph("<b>Best For</b>", styles["TH"])],
        [Paragraph("8", styles["TC"]), Paragraph("80 x 80 = 6,400 cells", styles["TC"]),
         Paragraph("8x8 pixel region", styles["TC"]), Paragraph("Small faces (20-80px) - CRITICAL for 7m distance", styles["TC"])],
        [Paragraph("16", styles["TC"]), Paragraph("40 x 40 = 1,600 cells", styles["TC"]),
         Paragraph("16x16 pixel region", styles["TC"]), Paragraph("Medium faces (80-200px)", styles["TC"])],
        [Paragraph("32", styles["TC"]), Paragraph("20 x 20 = 400 cells", styles["TC"]),
         Paragraph("32x32 pixel region", styles["TC"]), Paragraph("Large faces (200px+) - front row students", styles["TC"])],
    ]
    story.append(tbl(stride_data, [55, 130, 115, 150]))

    # 2.2 Anchors
    story.append(Paragraph("2.2. Multi-Scale Anchor System", styles["S2"]))
    story.append(Paragraph(
        "Each cell in each feature map has <b>2 anchors</b> (predefined reference boxes). The model doesn't "
        "predict absolute coordinates - it predicts <b>offsets</b> (deltas) relative to these anchors. "
        "Total anchors per frame: (6,400 + 1,600 + 400) x 2 = <b>16,800 anchor positions</b>.",
        styles["B"]))
    story.append(Paragraph(
        "For each anchor, the model predicts 3 things:",
        styles["B"]))
    story.append(Paragraph("- <b>1 score</b>: probability that this anchor contains a face (0 to 1)", styles["Bul"]))
    story.append(Paragraph("- <b>4 bbox deltas</b>: how to adjust the anchor to fit the actual face (dx1, dy1, dx2, dy2)", styles["Bul"]))
    story.append(Paragraph("- <b>10 landmark deltas</b>: offsets for 5 keypoints x 2 coordinates (x,y)", styles["Bul"]))

    story.append(Paragraph("<b>Converting anchors to actual coordinates:</b>", styles["S3"]))
    story.append(Paragraph(
        "Each anchor center is at the grid cell position multiplied by the stride. For example, "
        "grid cell (5, 3) at stride 8 has its anchor center at pixel (40, 24). The bounding box formula:",
        styles["B"]))
    story.append(Paragraph(
        "x1 = anchor_center_x - delta_left<br/>"
        "y1 = anchor_center_y - delta_top<br/>"
        "x2 = anchor_center_x + delta_right<br/>"
        "y2 = anchor_center_y + delta_bottom",
        styles["Math"]))
    story.append(Paragraph(
        "Similarly for landmarks, each keypoint is: kps_x = anchor_center_x + delta_x, kps_y = anchor_center_y + delta_y",
        styles["B"]))

    # 2.3 Loss
    story.append(Paragraph("2.3. Loss Functions and Training", styles["S2"]))
    story.append(Paragraph(
        "RetinaFace was trained using a multi-task loss function that jointly optimizes detection and landmarks:",
        styles["B"]))
    story.append(Paragraph(
        "L = L<sub>cls</sub> + lambda1 * L<sub>box</sub> + lambda2 * L<sub>pts</sub>",
        styles["Math"]))
    story.append(Paragraph(
        "<b>L<sub>cls</sub> (Classification Loss):</b> Binary cross-entropy loss that teaches the model to predict "
        "whether each anchor contains a face or not. Uses the <b>Focal Loss</b> variant which down-weights easy "
        "examples (obvious non-faces) and focuses training on hard cases (ambiguous regions). This is critical "
        "because 99% of anchors are background, not faces.",
        styles["B"]))
    story.append(Paragraph(
        "Focal Loss formula: FL(p) = -alpha * (1 - p)<super>gamma</super> * log(p), where gamma=2 focuses on hard examples.",
        styles["Math"]))
    story.append(Paragraph(
        "<b>L<sub>box</sub> (Bounding Box Regression Loss):</b> Smooth L1 loss between predicted box deltas and "
        "ground truth box positions. Teaches the model to precisely locate face boundaries.",
        styles["B"]))
    story.append(Paragraph(
        "<b>L<sub>pts</sub> (Landmark Loss):</b> Smooth L1 loss between predicted landmark deltas and ground truth "
        "5-point landmarks. This joint training improves both detection and alignment quality.",
        styles["B"]))

    # 2.4 NMS
    story.append(Paragraph("2.4. Post-Processing: Non-Maximum Suppression (NMS)", styles["S2"]))
    story.append(Paragraph(
        "After the model runs, we may have hundreds of overlapping detections for the same face (multiple "
        "anchors at multiple strides all detecting it). NMS removes these duplicates:",
        styles["B"]))
    story.append(Paragraph(
        "<b>NMS Algorithm Step by Step:</b>", styles["B"]))
    story.append(Paragraph("1. Sort all detections by confidence score (highest first)", styles["Bul"]))
    story.append(Paragraph("2. Take the highest-scoring detection - this is a keeper", styles["Bul"]))
    story.append(Paragraph("3. Compute IoU (Intersection over Union) between this box and ALL remaining boxes", styles["Bul"]))
    story.append(Paragraph("4. Remove any box with IoU > 0.4 (they overlap too much - they're detecting the same face)", styles["Bul"]))
    story.append(Paragraph("5. Repeat from step 2 with remaining boxes until none are left", styles["Bul"]))

    story.append(Paragraph(
        "<b>IoU (Intersection over Union) formula:</b>",
        styles["B"]))
    story.append(Paragraph(
        "IoU(A, B) = Area(A intersection B) / Area(A union B)<br/>"
        "= Area(overlap) / (Area(A) + Area(B) - Area(overlap))",
        styles["Math"]))
    story.append(Paragraph(
        "If two boxes have IoU > 0.4, they are considered to be detecting the same face, and the "
        "lower-confidence one is removed. This ensures we get exactly one detection per face.",
        styles["B"]))

    # 2.5 Our implementation
    story.append(Paragraph("2.5. Our Implementation in detector.py", styles["S2"]))
    story.append(Paragraph("<b>_preprocess_det(img):</b>", styles["S3"]))
    story.append(Paragraph(
        "Takes the raw 720p BGR camera frame and prepares it for the model. Steps: (1) Calculate scale factor "
        "to fit 1280x720 into 640x640 while keeping aspect ratio. (2) Resize image. (3) Create a black 640x640 "
        "canvas and paste the resized image (letterboxing). (4) Normalize pixel values from [0,255] to [-1,1] "
        "using formula: (pixel - 127.5) / 128.0. (5) Transpose from HWC (height, width, channels) to CHW format "
        "(channels first, as the model expects). (6) Add batch dimension: shape becomes (1, 3, 640, 640).",
        styles["B"]))

    story.append(Paragraph("<b>_postprocess_det(outputs, scale):</b>", styles["S3"]))
    story.append(Paragraph(
        "Parses the 9 output tensors from RetinaFace (3 strides x 3 outputs each). For each stride: "
        "(1) Generate anchor center coordinates for the feature map grid. (2) Filter anchors by score threshold "
        "(default 0.5). (3) Convert deltas to absolute pixel coordinates using _distance2bbox() and _distance2kps(). "
        "(4) Scale coordinates back to original image size (divide by the resize scale). After processing all "
        "strides, concatenate results and apply NMS.",
        styles["B"]))

    story.append(Paragraph("<b>_extract_embedding(img, face):</b>", styles["S3"]))
    story.append(Paragraph(
        "Takes the original image and a detected Face object. (1) Calls _align_face() to warp the face to "
        "112x112 using landmarks. (2) Normalizes pixels to [-1, 1]. (3) Transposes to CHW and adds batch dim. "
        "(4) Runs the ArcFace ONNX model. (5) L2-normalizes the output to unit length. Returns a 512-dim vector.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 3. ARCFACE =====
    story.append(Paragraph("3. ArcFace - Face Recognition Model", styles["S1"]))
    story.append(Paragraph(
        "ArcFace (Deng et al., 2019, CVPR) is a face recognition model that learns to map face images to a "
        "512-dimensional embedding space where same-person faces are close together and different-person faces "
        "are far apart. Our model file w600k_r50.onnx was trained on 600,000 different identities.",
        styles["B"]))

    # 3.1 ResNet-50
    story.append(Paragraph("3.1. ResNet-50 Backbone Architecture", styles["S2"]))
    story.append(Paragraph(
        "The backbone is a ResNet-50 (Residual Network with 50 layers). Key concepts:",
        styles["B"]))
    story.append(Paragraph(
        "<b>Residual Connections (Skip Connections):</b> The core innovation of ResNet. Instead of learning "
        "the direct mapping H(x), each block learns the <b>residual</b> F(x) = H(x) - x. The output is "
        "F(x) + x. This means if a layer has nothing useful to add, F(x) can be zero and the input passes "
        "through unchanged. This solves the vanishing gradient problem and allows training very deep networks.",
        styles["B"]))
    story.append(Paragraph(
        "Residual Block: output = F(x) + x, where F(x) = W2 * ReLU(W1 * x)",
        styles["Math"]))
    story.append(Paragraph(
        "<b>Network structure:</b> Input 112x112x3 -> Conv 3x3 (64 filters) -> 4 residual stages with "
        "[3, 4, 6, 3] blocks each -> Global Average Pooling -> Fully Connected -> 512-dim output. "
        "Total: ~25.5 million parameters. The output is a raw 512-dim vector that gets L2-normalized.",
        styles["B"]))

    # 3.2 ArcFace Loss
    story.append(Paragraph("3.2. ArcFace Loss Function (The Core Mathematics)", styles["S2"]))
    story.append(Paragraph(
        "This is the key innovation that makes ArcFace achieve 99.4% accuracy. Understanding it requires "
        "building up from the standard softmax loss.",
        styles["B"]))

    story.append(Paragraph("<b>Step 1: Standard Softmax Loss</b>", styles["S3"]))
    story.append(Paragraph(
        "In a standard classifier, the loss is:",
        styles["B"]))
    story.append(Paragraph(
        "L = -log( e<super>W_yi * x_i + b_yi</super> / SUM_j(e<super>W_j * x_i + b_j</super>) )",
        styles["Math"]))
    story.append(Paragraph(
        "Where W_yi is the weight vector for the correct class, x_i is the face embedding, and the sum is "
        "over all classes (identities). This works but doesn't enforce a clear margin between classes.",
        styles["B"]))

    story.append(Paragraph("<b>Step 2: Normalize Weights and Features</b>", styles["S3"]))
    story.append(Paragraph(
        "ArcFace normalizes both the weight vectors ||W_j|| = 1 and the feature vectors ||x_i|| = 1. "
        "After normalization, the dot product W_j * x_i = cos(theta_j), where theta_j is the angle between "
        "the feature vector and the class center on a hypersphere.",
        styles["B"]))
    story.append(Paragraph(
        "W_j * x_i = ||W_j|| * ||x_i|| * cos(theta_j) = cos(theta_j)  (after normalization)",
        styles["Math"]))

    story.append(Paragraph("<b>Step 3: Add Angular Margin (The ArcFace Innovation)</b>", styles["S3"]))
    story.append(Paragraph(
        "ArcFace adds a fixed margin <b>m</b> to the angle between the feature and its correct class center:",
        styles["B"]))
    story.append(Paragraph(
        "L = -log( e<super>s * cos(theta_yi + m)</super> / (e<super>s * cos(theta_yi + m)</super> + SUM_j!=yi(e<super>s * cos(theta_j)</super>)) )",
        styles["Math"]))
    story.append(Paragraph(
        "Where <b>s</b> = 64 (scale factor) and <b>m</b> = 0.5 radians (~28.6 degrees) are hyperparameters.",
        styles["B"]))
    story.append(Paragraph(
        "<b>What does this mean geometrically?</b> To correctly classify a face, the model must push the "
        "embedding vector to be at least <b>m = 0.5 radians closer</b> to its correct class center than to any "
        "other class center. This creates a clear 'decision boundary gap' on the hypersphere between all identities. "
        "The result: same-person embeddings are tightly clustered, and different-person embeddings are pushed far apart.",
        styles["B"]))

    story.append(Paragraph(
        "<b>Why 'Arc' Face?</b> Because the margin <b>m</b> is added to the <b>arc</b> (angle) in the cosine function. "
        "This is geometrically equivalent to enforcing a geodesic distance margin on the hypersphere surface.",
        styles["B"]))

    # 3.3 Embedding space
    story.append(Paragraph("3.3. Embedding Space and Cosine Similarity", styles["S2"]))
    story.append(Paragraph(
        "At inference time (our code), we don't use the classification head. We extract the 512-dim vector "
        "before the final classification layer, L2-normalize it, and use cosine similarity for matching.",
        styles["B"]))
    story.append(Paragraph(
        "<b>Cosine Similarity formula:</b>",
        styles["B"]))
    story.append(Paragraph(
        "similarity(A, B) = (A . B) / (||A|| * ||B||) = SUM(A_i * B_i) / (sqrt(SUM(A_i<super>2</super>)) * sqrt(SUM(B_i<super>2</super>)))",
        styles["Math"]))
    story.append(Paragraph(
        "Since our vectors are already L2-normalized (||A|| = ||B|| = 1), this simplifies to:",
        styles["B"]))
    story.append(Paragraph(
        "similarity(A, B) = A . B = SUM(A_i * B_i)    (just a dot product!)",
        styles["Math"]))
    story.append(Paragraph(
        "The result ranges from -1 (completely different) to +1 (identical). In practice, same-person "
        "pairs typically score 0.5-0.8, while different-person pairs score 0.0-0.3. Our threshold of 0.4 "
        "sits in the gap between these distributions.",
        styles["B"]))

    score_data = [
        [Paragraph("<b>Score Range</b>", styles["TH"]), Paragraph("<b>Meaning</b>", styles["TH"]),
         Paragraph("<b>Our System Action</b>", styles["TH"])],
        [Paragraph("0.4 - 1.0", styles["TC"]), Paragraph("Strong match - same person", styles["TC"]),
         Paragraph("Mark student as PRESENT", styles["TC"])],
        [Paragraph("0.3 - 0.4", styles["TC"]), Paragraph("Uncertain - might be same person", styles["TC"]),
         Paragraph("Mark as unknown, save face image for review", styles["TC"])],
        [Paragraph("0.0 - 0.3", styles["TC"]), Paragraph("Different person or not enrolled", styles["TC"]),
         Paragraph("Mark as unknown, save face image", styles["TC"])],
    ]
    story.append(tbl(score_data, [80, 200, 170]))

    # 3.4 Face Alignment
    story.append(Paragraph("3.4. Face Alignment (Similarity Transform)", styles["S2"]))
    story.append(Paragraph(
        "Before feeding a face to ArcFace, it MUST be aligned. This is one of the most important steps and "
        "directly affects recognition accuracy. Here's why and how:",
        styles["B"]))
    story.append(Paragraph(
        "<b>Why align?</b> ArcFace was trained on aligned faces where eyes are always at the same pixel "
        "positions. If we feed it a tilted or shifted face, the features won't match the training distribution "
        "and accuracy drops dramatically.",
        styles["B"]))
    story.append(Paragraph(
        "<b>The 5 reference points (ArcFace standard for 112x112):</b>",
        styles["B"]))

    ref_data = [
        [Paragraph("<b>Landmark</b>", styles["TH"]), Paragraph("<b>X</b>", styles["TH"]),
         Paragraph("<b>Y</b>", styles["TH"])],
        [Paragraph("Left Eye", styles["TC"]), Paragraph("38.29", styles["TC"]), Paragraph("51.70", styles["TC"])],
        [Paragraph("Right Eye", styles["TC"]), Paragraph("73.53", styles["TC"]), Paragraph("51.50", styles["TC"])],
        [Paragraph("Nose Tip", styles["TC"]), Paragraph("56.03", styles["TC"]), Paragraph("71.74", styles["TC"])],
        [Paragraph("Left Mouth", styles["TC"]), Paragraph("41.55", styles["TC"]), Paragraph("92.37", styles["TC"])],
        [Paragraph("Right Mouth", styles["TC"]), Paragraph("70.73", styles["TC"]), Paragraph("92.20", styles["TC"])],
    ]
    story.append(tbl(ref_data, [100, 80, 80]))

    story.append(Paragraph(
        "<b>Similarity Transform:</b> We compute a 2x3 affine transformation matrix that maps the detected "
        "landmarks to these reference positions. The transform handles rotation, scaling, and translation. "
        "We use OpenCV's cv2.estimateAffinePartial2D() which finds the best fit using least squares:",
        styles["B"]))
    story.append(Paragraph(
        "[x'] = [a  -b  tx] [x]<br/>"
        "[y']   [b   a  ty] [y]<br/>"
        "Where a,b encode rotation+scale, tx,ty encode translation",
        styles["Math"]))
    story.append(Paragraph(
        "The image is then warped using cv2.warpAffine() to produce a clean 112x112 aligned face where "
        "the eyes, nose, and mouth are in standardized positions regardless of the original head pose.",
        styles["B"]))
    story.append(PageBreak())

    # 3.5 Our implementation
    story.append(Paragraph("3.5. Our Implementation in recognizer.py", styles["S2"]))
    story.append(Paragraph(
        "<b>Enrollment storage format:</b> Each student gets a directory: data/embeddings/{student_id}/ "
        "containing embeddings.npy (Nx512 numpy array, one row per photo) and name.txt.",
        styles["B"]))
    story.append(Paragraph(
        "<b>Matching algorithm:</b> For a query embedding Q against student S who has N enrolled embeddings:",
        styles["B"]))
    story.append(Paragraph(
        "1. Compute dot product: similarities = S.embeddings @ Q  (result: N scores)<br/>"
        "2. Take max: score = max(similarities)<br/>"
        "3. Compare score against threshold (0.4)<br/>"
        "4. Return best matching student across all enrolled students",
        styles["Math"]))
    story.append(Paragraph(
        "Using the maximum across multiple enrollment embeddings is key - if a student was enrolled looking "
        "left and looking right, we match against whichever pose is closest to their current pose.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 4. DIFFICULT CONDITIONS =====
    story.append(Paragraph("4. How We Handle Difficult Conditions", styles["S1"]))

    story.append(Paragraph("4.1. Poor Lighting and Shadows", styles["S2"]))
    story.append(Paragraph(
        "<b>How ArcFace handles this:</b> During training on 600K identities, the model saw faces under "
        "every imaginable lighting condition - studio lighting, outdoor sun, indoor fluorescent, side lighting, "
        "backlighting, etc. The L2 normalization of embeddings helps because it removes the effect of overall "
        "brightness (a dark photo and a bright photo of the same face will have similar normalized embeddings). "
        "The angular margin loss further forces the model to focus on identity-specific features rather than "
        "lighting-specific features.",
        styles["B"]))
    story.append(Paragraph(
        "<b>What helps in our preprocessing:</b> The normalization step (pixel - 127.5) / 128.0 centers the "
        "pixel values around zero, which partially compensates for overall brightness differences. The face "
        "alignment also helps because it standardizes the face position, meaning shadow patterns are more consistent.",
        styles["B"]))
    story.append(Paragraph(
        "<b>Limitation:</b> Extremely dim conditions where the camera itself can't capture useful detail "
        "will still fail. The model can't see what the camera can't capture.",
        styles["B"]))

    story.append(Paragraph("4.2. Angled Camera (Non-Frontal Faces)", styles["S2"]))
    story.append(Paragraph(
        "The camera is mounted at a corner or above the whiteboard, creating an angled view. This is handled "
        "at multiple levels:",
        styles["B"]))
    story.append(Paragraph(
        "<b>1. Detection (RetinaFace):</b> RetinaFace can detect faces at angles up to about 60-70 degrees from "
        "frontal. The multi-scale FPN helps because angled faces appear at different sizes depending on distance.",
        styles["B"]))
    story.append(Paragraph(
        "<b>2. Alignment (The Key Step):</b> This is WHERE the angle problem is solved. Even if the face is "
        "detected at an angle, the 5-point landmarks tell us exactly where the eyes, nose, and mouth are in "
        "the rotated/perspected view. The similarity transform then warps the face as if it were frontal. "
        "The landmarks 'undo' the rotation by mapping the detected positions back to the standard template. "
        "This is why landmark-based alignment is critical for our scenario.",
        styles["B"]))
    story.append(Paragraph(
        "<b>3. Multi-pose enrollment:</b> We take 5 photos at different angles during enrollment (front, left, "
        "right). Even if alignment doesn't perfectly undo the angle, one of the enrolled poses will be close "
        "to the current viewing angle, giving a high similarity score.",
        styles["B"]))
    story.append(Paragraph(
        "<b>4. Recognition (ArcFace):</b> Trained on faces with pose variation up to ~45 degrees. The angular "
        "margin loss encourages pose-invariant features. Combined with alignment, this handles most classroom angles.",
        styles["B"]))

    story.append(Paragraph("4.3. Small and Distant Faces (7 meters)", styles["S2"]))
    story.append(Paragraph(
        "At 7 meters on a 720p camera, a face may be only 30-50 pixels wide. This is handled by:",
        styles["B"]))
    story.append(Paragraph(
        "<b>RetinaFace stride-8 feature map:</b> This is the highest resolution feature map (80x80 for 640x640 input). "
        "Each cell covers only 8x8 pixels, allowing detection of faces as small as ~20 pixels. Without this "
        "stride, small faces would be invisible to the model. This is the primary reason we chose RetinaFace "
        "over YOLO - RetinaFace's FPN with stride-8 is specifically designed for tiny face detection.",
        styles["B"]))
    story.append(Paragraph(
        "<b>The 640x640 detection resolution:</b> We resize the 1280x720 input to fit a 640x640 canvas. "
        "If detection accuracy is insufficient, this can be increased to 960x960 or 1280x1280 in config.yaml "
        "(at the cost of more GPU memory and slower inference).",
        styles["B"]))
    story.append(Paragraph(
        "<b>Limitation:</b> After detection, the face is aligned to 112x112 for recognition. If the original "
        "face was only 30px wide, upscaling to 112px introduces blur. Recognition accuracy will be lower for "
        "very distant faces. This is a fundamental limitation that can only be solved with a higher resolution camera.",
        styles["B"]))

    story.append(Paragraph("4.4. Many Faces (30-50 Students)", styles["S2"]))
    story.append(Paragraph(
        "RetinaFace processes the entire frame in one forward pass regardless of face count - it checks all "
        "16,800 anchors simultaneously. The max_faces=60 config limits post-NMS output. Recognition runs "
        "sequentially per face (~1ms each on GPU), so 50 faces take ~50ms total. The 10-second capture "
        "interval means total GPU time is ~80ms every 10 seconds = less than 1% GPU utilization.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 5. FINE-TUNING =====
    story.append(Paragraph("5. Fine-Tuning Guide", styles["S1"]))

    story.append(Paragraph("5.1. Can We Fine-Tune? What and Why?", styles["S2"]))
    story.append(Paragraph(
        "<b>Yes, both models can be fine-tuned.</b> However, the strategy differs:",
        styles["B"]))

    ft_data = [
        [Paragraph("<b>Model</b>", styles["TH"]), Paragraph("<b>Fine-Tune?</b>", styles["TH"]),
         Paragraph("<b>When</b>", styles["TH"]), Paragraph("<b>Expected Benefit</b>", styles["TH"])],
        [Paragraph("RetinaFace (detection)", styles["TC"]),
         Paragraph("Usually NOT needed", styles["TC"]),
         Paragraph("Only if detection fails in your specific lighting/angle", styles["TC"]),
         Paragraph("Better detection rate in your classroom", styles["TC"])],
        [Paragraph("ArcFace (recognition)", styles["TC"]),
         Paragraph("YES - recommended", styles["TC"]),
         Paragraph("After initial testing shows recognition errors", styles["TC"]),
         Paragraph("5-10% accuracy improvement on your student population", styles["TC"])],
    ]
    story.append(tbl(ft_data, [100, 80, 140, 130]))

    story.append(Paragraph(
        "<b>Why fine-tune ArcFace?</b> The pretrained model was trained on 600K mostly Western/East Asian faces. "
        "If your student population has different ethnic/cultural features, fine-tuning on your actual students "
        "teaches the model to focus on the distinguishing features of YOUR population.",
        styles["B"]))

    story.append(Paragraph("5.2. Datasets for Fine-Tuning", styles["S2"]))

    story.append(Paragraph("<b>Public Datasets (for pre-training/augmentation):</b>", styles["S3"]))
    ds_data = [
        [Paragraph("<b>Dataset</b>", styles["TH"]), Paragraph("<b>Size</b>", styles["TH"]),
         Paragraph("<b>Use Case</b>", styles["TH"]), Paragraph("<b>Source</b>", styles["TH"])],
        [Paragraph("MS1MV2", styles["TC"]), Paragraph("5.8M images, 85K IDs", styles["TC"]),
         Paragraph("Large-scale pre-training", styles["TC"]),
         Paragraph("insightface/datasets", styles["TC"])],
        [Paragraph("VGGFace2", styles["TC"]), Paragraph("3.3M images, 9.1K IDs", styles["TC"]),
         Paragraph("Diverse ages/ethnicities", styles["TC"]),
         Paragraph("robots.ox.ac.uk", styles["TC"])],
        [Paragraph("LFW", styles["TC"]), Paragraph("13K images, 5.7K IDs", styles["TC"]),
         Paragraph("Benchmark testing only", styles["TC"]),
         Paragraph("vis-www.cs.umass.edu", styles["TC"])],
        [Paragraph("AgeDB-30", styles["TC"]), Paragraph("16K images", styles["TC"]),
         Paragraph("Age-variation testing", styles["TC"]),
         Paragraph("github.com/AgeDB", styles["TC"])],
        [Paragraph("WIDER FACE", styles["TC"]), Paragraph("32K images, 393K faces", styles["TC"]),
         Paragraph("Detection fine-tuning", styles["TC"]),
         Paragraph("shuoyang1213.me", styles["TC"])],
    ]
    story.append(tbl(ds_data, [70, 110, 120, 140]))

    story.append(Paragraph("<b>Custom Student Dataset (MOST IMPORTANT):</b>", styles["S3"]))
    story.append(Paragraph(
        "The most impactful fine-tuning uses photos of your actual students. Recommended collection protocol:",
        styles["B"]))
    story.append(Paragraph("- 10-20 photos per student (more is better)", styles["Bul"]))
    story.append(Paragraph("- Taken in the ACTUAL classroom with the ACTUAL camera", styles["Bul"]))
    story.append(Paragraph("- Vary poses: front, left 15 deg, right 15 deg, slight up, slight down", styles["Bul"]))
    story.append(Paragraph("- Vary conditions: morning light, afternoon light, curtains open/closed", styles["Bul"]))
    story.append(Paragraph("- Include glasses on/off if applicable", styles["Bul"]))
    story.append(Paragraph("- Minimum 50 students for meaningful fine-tuning (more is better)", styles["Bul"]))

    story.append(Paragraph("5.3. Fine-Tuning Strategy", styles["S2"]))
    story.append(Paragraph(
        "<b>Approach: Transfer Learning with ArcFace Loss</b>",
        styles["B"]))
    story.append(Paragraph(
        "1. Start with the pretrained w600k_r50 weights (our current model)<br/>"
        "2. Replace the classification head (600K classes -> your N students)<br/>"
        "3. Freeze early layers (they detect universal features like edges, textures)<br/>"
        "4. Train only the last few layers + new classification head<br/>"
        "5. Use ArcFace loss (s=64, m=0.5) - same as original training<br/>"
        "6. Train for 10-20 epochs with learning rate 0.001, reducing to 0.0001<br/>"
        "7. Export back to ONNX format for deployment",
        styles["B"]))
    story.append(Paragraph(
        "<b>Framework:</b> PyTorch with the insightface training scripts (github.com/deepinsight/insightface/tree/master/recognition).",
        styles["B"]))
    story.append(Paragraph(
        "<b>Note:</b> Fine-tuning is a LATER step. First test with the pretrained model, measure accuracy, "
        "then decide if fine-tuning is needed based on real-world performance.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 6. CODE WALKTHROUGH =====
    story.append(Paragraph("6. Complete Code Walkthrough", styles["S1"]))

    story.append(Paragraph("6.1. detector.py - Core AI Module", styles["S2"]))
    story.append(Paragraph(
        "This is the most important file. It contains all the AI logic for face detection and embedding extraction.",
        styles["B"]))

    story.append(Paragraph("<b>Face Dataclass (line 17-23):</b>", styles["S3"]))
    story.append(Paragraph(
        "A simple data container holding: bbox (bounding box as [x1,y1,x2,y2] numpy array), det_score (detection "
        "confidence 0-1), landmarks (5x2 numpy array of facial keypoints), and embedding (512-dim vector or None).",
        styles["B"]))

    story.append(Paragraph("<b>_distance2bbox() (line 26-32):</b>", styles["S3"]))
    story.append(Paragraph(
        "Converts anchor-relative distance predictions to absolute bounding box coordinates. Takes anchor "
        "center points and distance predictions [left, top, right, bottom], returns [x1, y1, x2, y2]. "
        "Pure vectorized numpy for speed.",
        styles["B"]))

    story.append(Paragraph("<b>_distance2kps() (line 35-42):</b>", styles["S3"]))
    story.append(Paragraph(
        "Same concept but for 5 keypoints. Takes anchor centers and 10 delta values, returns 5 (x,y) coordinates. "
        "Each keypoint is: x = anchor_x + delta_x, y = anchor_y + delta_y.",
        styles["B"]))

    story.append(Paragraph("<b>_nms() (line 45-71):</b>", styles["S3"]))
    story.append(Paragraph(
        "Non-Maximum Suppression implementation. Sorts by score, iteratively keeps the highest-scoring box "
        "and removes all boxes that overlap it by more than the IoU threshold (0.4). Uses numpy vectorized "
        "operations for efficiency.",
        styles["B"]))

    story.append(Paragraph("<b>_align_face() (line 74-93):</b>", styles["S3"]))
    story.append(Paragraph(
        "The critical face alignment function. Defines the ArcFace standard reference points for 112x112, "
        "then computes a similarity transform to warp the detected face landmarks onto these reference points. "
        "Uses cv2.warpAffine to produce the aligned 112x112 output. THIS is what handles the camera angle problem.",
        styles["B"]))

    story.append(Paragraph("<b>FaceDetector.__init__() (line 131-157):</b>", styles["S3"]))
    story.append(Paragraph(
        "Loads both ONNX models into ONNX Runtime inference sessions. Requests CUDA first, falls back to CPU. "
        "Prints which provider is active so you can verify GPU is being used.",
        styles["B"]))

    story.append(Paragraph("<b>FaceDetector.detect_faces() (line 277-291):</b>", styles["S3"]))
    story.append(Paragraph(
        "The main entry point called by main.py. Preprocesses the frame, runs detection, parses outputs, "
        "then extracts embeddings for each detected face. Returns a list of Face objects ready for matching.",
        styles["B"]))

    story.append(Paragraph("6.2. recognizer.py - Embedding Database", styles["S2"]))
    story.append(Paragraph(
        "<b>load_database():</b> Scans data/embeddings/ directory. Each subdirectory is a student ID. Loads "
        "embeddings.npy (Nx512) and name.txt into memory. Called once at startup.",
        styles["B"]))
    story.append(Paragraph(
        "<b>enroll_student():</b> Takes a student ID, name, and list of 512-dim embeddings. Stacks them into "
        "a numpy array, saves to disk as .npy file. Also updates the in-memory database.",
        styles["B"]))
    story.append(Paragraph(
        "<b>recognize():</b> The matching function. For each enrolled student, computes cosine similarity "
        "between the query and ALL of that student's enrolled embeddings (vectorized with numpy). Takes the "
        "max similarity across all enrollments. Returns the best matching student above the threshold.",
        styles["B"]))

    story.append(Paragraph("6.3. attendance_manager.py", styles["S2"]))
    story.append(Paragraph(
        "<b>mark_detected():</b> Called when a face is matched. If the student hasn't been seen before, creates "
        "a new record with first_seen timestamp. If already seen, updates last_seen and increments detection_count. "
        "Also tracks best_confidence for reporting.",
        styles["B"]))
    story.append(Paragraph(
        "<b>get_full_report():</b> Compares detected students against the expected roster (from backend API) "
        "to determine who is absent. Returns a dict with present list, absent list, unknown count, and metadata.",
        styles["B"]))

    story.append(Paragraph("6.4. main.py - The Attendance Loop", styles["S2"]))
    story.append(Paragraph(
        "The main loop runs continuously until the configured duration expires:",
        styles["B"]))
    story.append(Paragraph(
        "1. Read frame from camera (every iteration, for smooth display)<br/>"
        "2. Check if capture_interval has elapsed (default 10 seconds)<br/>"
        "3. If yes: run face detection -> recognition -> attendance marking<br/>"
        "4. If no: just display the frame with previous annotations<br/>"
        "5. Calculate and display FPS, time remaining, present count<br/>"
        "6. Check for 'Q' key press to stop early<br/>"
        "7. On exit: generate report, save CSV, submit to API",
        styles["B"]))

    story.append(Paragraph("6.5. enroll.py - Enrollment Flow", styles["S2"]))
    story.append(Paragraph(
        "Interactive script that opens the webcam and guides the user through capturing 5 photos at different "
        "poses. Uses find_largest_face() to select the target face (the person standing closest to the camera). "
        "Requires minimum 3 successful captures. Stores embeddings to disk via FaceRecognizer.enroll_student().",
        styles["B"]))
    story.append(PageBreak())

    # ===== 7. HOW TO RUN =====
    story.append(Paragraph("7. How to Run the System", styles["S1"]))

    story.append(Paragraph("<b>Prerequisites:</b>", styles["S2"]))
    story.append(Paragraph(
        "1. Python 3.10 virtual environment: C:\\Users\\abdul\\attend_venv<br/>"
        "2. ONNX models in attendance_system/models/ (det_10g.onnx + w600k_r50.onnx)<br/>"
        "3. USB webcam connected",
        styles["B"]))

    story.append(Paragraph("<b>Step 1: Enroll Students</b>", styles["S2"]))
    story.append(Paragraph(
        "Run from the attendance_system directory:",
        styles["B"]))
    story.append(Paragraph(
        'C:\\Users\\abdul\\attend_venv\\Scripts\\python.exe enroll.py --student-id "12345" --student-name "Ahmed Ali"',
        styles["CodeBlock"]))
    story.append(Paragraph(
        "A window opens showing the webcam feed. Follow the on-screen prompts for each pose. "
        "Press SPACE to capture each photo. Press Q to cancel. Repeat for each student.",
        styles["B"]))

    story.append(Paragraph("<b>Step 2: Run Attendance</b>", styles["S2"]))
    story.append(Paragraph(
        'C:\\Users\\abdul\\attend_venv\\Scripts\\python.exe main.py --duration 60 --interval 10 --class-id "CS101"',
        styles["CodeBlock"]))

    cmd_data = [
        [Paragraph("<b>Flag</b>", styles["TH"]), Paragraph("<b>Default</b>", styles["TH"]),
         Paragraph("<b>Description</b>", styles["TH"])],
        [Paragraph("--duration", styles["TC"]), Paragraph("60", styles["TC"]),
         Paragraph("How many minutes to run", styles["TC"])],
        [Paragraph("--interval", styles["TC"]), Paragraph("10", styles["TC"]),
         Paragraph("Seconds between face captures", styles["TC"])],
        [Paragraph("--class-id", styles["TC"]), Paragraph("None", styles["TC"]),
         Paragraph("Tag the report with a class ID", styles["TC"])],
        [Paragraph("--no-display", styles["TC"]), Paragraph("False", styles["TC"]),
         Paragraph("Run headless without GUI window", styles["TC"])],
    ]
    story.append(tbl(cmd_data, [80, 60, 310]))

    story.append(Paragraph("<b>Step 3: View Results</b>", styles["S2"]))
    story.append(Paragraph(
        "After the session ends, the system prints a summary and saves a CSV file to "
        "data/attendance_csv/attendance_YYYY-MM-DD.csv with columns: student_id, student_name, status, "
        "first_seen, last_seen, best_confidence, detection_count.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 8. API REFERENCE =====
    story.append(Paragraph("8. API Reference (Backend Integration)", styles["S1"]))
    story.append(Paragraph(
        "The system communicates with the school management backend via REST API. "
        "If the backend is not running, the system works in offline mode (CSV only).",
        styles["B"]))

    story.append(Paragraph("8.1. What You Give the Backend (AI -> Backend)", styles["S2"]))

    story.append(Paragraph("<b>POST /api/attendance</b> - Submit attendance report", styles["S3"]))
    story.append(Paragraph("Sent automatically at the end of each attendance session.", styles["B"]))
    story.append(Paragraph(
        '{\n'
        '  "date": "2026-04-13",\n'
        '  "generated_at": "2026-04-13T08:05:23",\n'
        '  "class_id": "CS101",\n'
        '  "total_present": 28,\n'
        '  "total_absent": 4,\n'
        '  "total_unknown": 2,\n'
        '  "present": [\n'
        '    {\n'
        '      "student_id": "12345",\n'
        '      "student_name": "Ahmed Ali",\n'
        '      "status": "present",\n'
        '      "first_seen": "2026-04-13T07:02:15",\n'
        '      "last_seen": "2026-04-13T07:58:30",\n'
        '      "best_confidence": 0.72,\n'
        '      "detection_count": 35\n'
        '    }\n'
        '  ],\n'
        '  "absent": [\n'
        '    {\n'
        '      "student_id": "12346",\n'
        '      "student_name": "Sara Khan",\n'
        '      "status": "absent"\n'
        '    }\n'
        '  ]\n'
        '}',
        styles["CodeBlock"]))

    story.append(Paragraph("<b>POST /api/enroll</b> - Notify enrollment", styles["S3"]))
    story.append(Paragraph(
        '{"student_id": "12345", "student_name": "Ahmed Ali"}',
        styles["CodeBlock"]))

    story.append(Paragraph("8.2. What You Get from the Backend (Backend -> AI)", styles["S2"]))

    story.append(Paragraph("<b>GET /api/students</b> - Student roster", styles["S3"]))
    story.append(Paragraph("Called on startup to know which students should be in class.", styles["B"]))
    story.append(Paragraph(
        '[\n'
        '  {"student_id": "12345", "student_name": "Ahmed Ali"},\n'
        '  {"student_id": "12346", "student_name": "Sara Khan"},\n'
        '  ...\n'
        ']',
        styles["CodeBlock"]))

    story.append(Paragraph("<b>GET /api/schedule?date=2026-04-13</b> - Class schedule", styles["S3"]))
    story.append(Paragraph(
        '[\n'
        '  {\n'
        '    "class_id": "CS101",\n'
        '    "room_id": "A201",\n'
        '    "start_time": "07:00",\n'
        '    "end_time": "08:00",\n'
        '    "student_ids": ["12345", "12346", "12347"]\n'
        '  }\n'
        ']',
        styles["CodeBlock"]))

    story.append(Paragraph("8.3. Backend Team Instructions", styles["S2"]))
    story.append(Paragraph(
        "Tell your backend team to implement these 4 endpoints. The AI system makes standard HTTP requests "
        "with JSON bodies. The base URL is configured in config.yaml (default: http://localhost:8000/api). "
        "All responses should be JSON. If the backend returns non-200 status codes, the AI system logs a "
        "warning and continues in offline mode.",
        styles["B"]))
    story.append(PageBreak())

    # ===== 9. CONFIGURATION =====
    story.append(Paragraph("9. Configuration Reference", styles["S1"]))
    story.append(Paragraph("All settings are in config.yaml. Edit this file to tune the system.", styles["B"]))

    config_data = [
        [Paragraph("<b>Parameter</b>", styles["TH"]), Paragraph("<b>Default</b>", styles["TH"]),
         Paragraph("<b>What It Does</b>", styles["TH"]), Paragraph("<b>When to Change</b>", styles["TH"])],
        [Paragraph("camera.source", styles["TC"]), Paragraph("0", styles["TC"]),
         Paragraph("Camera input (0=USB, or RTSP URL)", styles["TC"]),
         Paragraph("When using IP camera", styles["TC"])],
        [Paragraph("det_thresh", styles["TC"]), Paragraph("0.5", styles["TC"]),
         Paragraph("Min confidence to consider a detection", styles["TC"]),
         Paragraph("Lower if missing distant faces; raise if getting false detections", styles["TC"])],
        [Paragraph("det_size", styles["TC"]), Paragraph("[640,640]", styles["TC"]),
         Paragraph("Resolution fed to RetinaFace", styles["TC"]),
         Paragraph("Increase for better small-face detection (uses more VRAM)", styles["TC"])],
        [Paragraph("similarity_threshold", styles["TC"]), Paragraph("0.4", styles["TC"]),
         Paragraph("Min cosine sim for positive ID", styles["TC"]),
         Paragraph("Raise if getting wrong IDs; lower if missing correct IDs", styles["TC"])],
        [Paragraph("unknown_threshold", styles["TC"]), Paragraph("0.3", styles["TC"]),
         Paragraph("Below this = definitely unknown", styles["TC"]),
         Paragraph("Adjust gap between this and similarity_threshold", styles["TC"])],
        [Paragraph("capture_interval_sec", styles["TC"]), Paragraph("10", styles["TC"]),
         Paragraph("How often to process faces", styles["TC"]),
         Paragraph("Lower for faster attendance; higher to save GPU", styles["TC"])],
        [Paragraph("session_duration_min", styles["TC"]), Paragraph("60", styles["TC"]),
         Paragraph("Total runtime", styles["TC"]),
         Paragraph("Match to your attendance period", styles["TC"])],
        [Paragraph("max_faces", styles["TC"]), Paragraph("60", styles["TC"]),
         Paragraph("Max faces per frame", styles["TC"]),
         Paragraph("Set above your largest class size", styles["TC"])],
        [Paragraph("photos_per_student", styles["TC"]), Paragraph("5", styles["TC"]),
         Paragraph("Enrollment photos count", styles["TC"]),
         Paragraph("More = better accuracy, longer enrollment", styles["TC"])],
        [Paragraph("min_confidence", styles["TC"]), Paragraph("0.4", styles["TC"]),
         Paragraph("Min confidence to mark present", styles["TC"]),
         Paragraph("Same as similarity_threshold usually", styles["TC"])],
    ]
    config_table = Table(config_data, colWidths=[90, 50, 140, 170])
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

    # ===== 10. LIMITATIONS =====
    story.append(Paragraph("10. Limitations and When It May Fail", styles["S1"]))

    lim_data = [
        [Paragraph("<b>Scenario</b>", styles["TH"]), Paragraph("<b>Problem</b>", styles["TH"]),
         Paragraph("<b>Severity</b>", styles["TH"]), Paragraph("<b>Mitigation</b>", styles["TH"])],
        [Paragraph("Face masks", styles["TC"]),
         Paragraph("Covers nose + mouth, removing 3 of 5 landmarks. Alignment fails, recognition drops severely.", styles["TC"]),
         Paragraph("HIGH", styles["TC"]),
         Paragraph("Need mask-aware model or periocular (eye-only) recognition", styles["TC"])],
        [Paragraph("Very dim room", styles["TC"]),
         Paragraph("Camera can't capture detail. Model receives noise.", styles["TC"]),
         Paragraph("HIGH", styles["TC"]),
         Paragraph("Improve classroom lighting or use IR camera", styles["TC"])],
        [Paragraph("Student looking straight down", styles["TC"]),
         Paragraph("Face not visible to camera at all", styles["TC"]),
         Paragraph("MEDIUM", styles["TC"]),
         Paragraph("Multiple captures over time - student will look up eventually", styles["TC"])],
        [Paragraph("Identical twins", styles["TC"]),
         Paragraph("Embeddings may be very similar", styles["TC"]),
         Paragraph("LOW", styles["TC"]),
         Paragraph("Fine-tune on their specific differences", styles["TC"])],
        [Paragraph("Photo spoofing", styles["TC"]),
         Paragraph("Someone holds up a photo of a student", styles["TC"]),
         Paragraph("MEDIUM", styles["TC"]),
         Paragraph("Add liveness detection (future work)", styles["TC"])],
        [Paragraph("Extreme backlight", styles["TC"]),
         Paragraph("Face appears as dark silhouette", styles["TC"]),
         Paragraph("MEDIUM", styles["TC"]),
         Paragraph("Camera HDR mode, or move camera position", styles["TC"])],
        [Paragraph("1000+ students in DB", styles["TC"]),
         Paragraph("Linear search becomes slow", styles["TC"]),
         Paragraph("LOW", styles["TC"]),
         Paragraph("Switch to FAISS approximate nearest neighbor search", styles["TC"])],
    ]
    lim_table = Table(lim_data, colWidths=[85, 140, 55, 170])
    lim_table.setStyle(TableStyle([
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
    story.append(lim_table)

    story.append(Spacer(1, 30))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc")))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"<i>Report generated on {date.today().strftime('%B %d, %Y')}. "
        "School Management System - Senior Project. "
        "AI Attendance System v1.0 using RetinaFace + ArcFace on ONNX Runtime.</i>",
        styles["B"]))

    doc.build(story)
    print(f"\nPDF report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    build_report()
