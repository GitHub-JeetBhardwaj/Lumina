import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify

# ==================== FLASK APP SETUP ====================

app = Flask(__name__)
app.secret_key = "secret_key_for_flash_messages"

# ==================== IMAGE ENHANCEMENT LOGIC ====================

def enhance_image(image):
    """
    Replaces the deep learning model.
    Uses OpenCV to equalize the histogram of the Y (luminance) channel.
    """
    # Convert PIL Image to a NumPy array
    img = np.array(image)
    
    # Convert the image to YUV color space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Equalize the histogram of the Y channel (luminance)
    yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
    
    # Convert the YUV image back to RGB format
    enhanced_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    
    # Convert back to PIL Image so it works with the rest of the Flask app
    return Image.fromarray(enhanced_img)

def to_base64(pil_img):
    """Helper to convert PIL Image to base64 string for HTML rendering."""
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=85)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

# ==================== ROUTES ====================

@app.route('/', methods=['GET', 'POST'])
def index():
    # If using Query Params to switch modes
    mode = request.args.get('mode', 'image')
    return render_template('index.html', mode=mode)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index', mode='image'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index', mode='image'))
        
    if file:
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            
            # Use OpenCV logic instead of the PyTorch inference
            enhanced_pil = enhance_image(img)
            
            original_b64 = to_base64(img)
            enhanced_b64 = to_base64(enhanced_pil)
            
            return render_template('index.html', 
                                   mode='image',
                                   original_img=original_b64, 
                                   enhanced_img=enhanced_b64,
                                   filename=file.filename)
                                   
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index', mode='image'))
            
    return redirect(url_for('index', mode='image'))

@app.route('/live_video')
def live_video():
    return render_template('index.html', mode='video')

# Processes frames sent via AJAX/JS
@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json
        image_data = data['image']
        
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Open Image
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Enhance using OpenCV logic
        enhanced_pil = enhance_image(img)
        
        # Return Enhanced Base64
        enhanced_b64 = to_base64(enhanced_pil)
        return jsonify({'status': 'success', 'image': 'data:image/jpeg;base64,' + enhanced_b64})

    except Exception as e:
        print(f"Frame error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Server starting with OpenCV Enhancement Logic.")
    app.run(host='0.0.0.0', port=7860)