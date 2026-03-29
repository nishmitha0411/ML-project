# =========================
# 👥 MULTIPLE FACE SUPPORT
# =========================
def analyze_face_detection(results, img_array):
    """Process multiple faces in one image"""
    
    h, w, _ = img_array.shape
    all_results = []
    
    for i, detection in enumerate(results.detections):
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        w_box = int(bbox.width * w)
        h_box = int(bbox.height * h)
        
        # Draw rectangle for each face
        cv2.rectangle(img_array, (x, y), (x + w_box, y + h_box), (102, 126, 234), 3)
        
        # Add label
        cv2.putText(img_array, f"Face {i+1}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 126, 234), 2)
        
        all_results.append({
            'face_num': i+1,
            'bbox': (x, y, w_box, h_box)
        })
    
    return img_array, all_results