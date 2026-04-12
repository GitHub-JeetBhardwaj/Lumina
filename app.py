import os
import io
import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, flash, redirect, url_for, jsonify

# ==================== FLASK APP SETUP ====================

app = Flask(__name__)
app.secret_key = "secret_key_for_flash_messages"

# ==================== PYTORCH ARCHITECTURE ====================

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 256)
        self.up5 = UNetUp(512, 128)
        self.up6 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Sigmoid() # Outputs [0, 1]
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)

# ==================== MODEL INITIALIZATION ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "gen_epoch_50.pth" # Ensure this path is correct
IMAGE_SIZE = 256

print(f"Loading model from {MODEL_PATH} onto {DEVICE}...")

# Initialize and load weights
generator = Generator().to(DEVICE)
try:
    generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    generator.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}. PyTorch DL logic will fail.")

# Inference Transforms
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

# ==================== ENHANCEMENT LOGIC ====================

def enhance_image_dl(image):
    """Deep Learning Method: Passes input image through trained PyTorch Generator."""
    original_size = image.size 
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    output_tensor = output_tensor.squeeze(0).cpu()
    enhanced_pil = to_pil(output_tensor)
    
    # Resize back to original dimensions
    enhanced_pil = enhanced_pil.resize(original_size, Image.Resampling.LANCZOS)
    return enhanced_pil

def enhance_image_cv2(image):
    """Traditional Method: Uses OpenCV to equalize the histogram of the Y (luminance) channel."""
    img = np.array(image)
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    
    # Equalize the histogram of the Y channel (luminance)
    yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
    
    enhanced_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2RGB)
    return Image.fromarray(enhanced_img)

def to_base64(pil_img):
    """Helper to convert PIL Image to base64 string for HTML rendering."""
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=85)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

# ==================== ROUTES ====================

@app.route('/', methods=['GET', 'POST'])
def index():
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
            
            # Run BOTH logic pathways
            cv2_enhanced_pil = enhance_image_cv2(img)
            dl_enhanced_pil = enhance_image_dl(img)
            
            # Convert to base64
            original_b64 = to_base64(img)
            cv2_b64 = to_base64(cv2_enhanced_pil)
            dl_b64 = to_base64(dl_enhanced_pil)
            
            return render_template('index.html', 
                                   mode='image',
                                   original_img=original_b64, 
                                   cv2_img=cv2_b64,       # Passed to Jinja template
                                   dl_img=dl_b64,         # Passed to Jinja template
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
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Enhance using BOTH methods
        cv2_enhanced_pil = enhance_image_cv2(img)
        dl_enhanced_pil = enhance_image_dl(img)
        
        # Return Both Enhanced Base64s
        cv2_b64 = to_base64(cv2_enhanced_pil)
        dl_b64 = to_base64(dl_enhanced_pil)
        
        return jsonify({
            'status': 'success', 
            'cv2_image': 'data:image/jpeg;base64,' + cv2_b64,
            'dl_image': 'data:image/jpeg;base64,' + dl_b64
        })

    except Exception as e:
        print(f"Frame error: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Server starting with BOTH OpenCV & PyTorch DL Logic...")
    app.run(host='0.0.0.0', port=7860)