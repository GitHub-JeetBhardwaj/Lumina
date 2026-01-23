import os
import io
import base64
import cv2  # NEW: OpenCV for video handling
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, flash, redirect, Response, url_for

# ==================== MODEL ARCHITECTURE ====================
# (Kept exactly as provided)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out

class SwinIRDenoiser(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, num_blocks=8):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.res_blocks = nn.ModuleList([ResidualBlock(embed_dim) for _ in range(num_blocks)])
        self.channel_attn = ChannelAttention(embed_dim)
        self.spatial_attn = SpatialAttention()
        self.conv_last = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)
    
    def forward(self, x):
        x = self.conv_first(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        x = self.conv_last(x)
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1, 1, 0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = self.conv_block(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = self.conv_block(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, 1)
    
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.up4(b)
        e4 = self.att4(d4, e4)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        e3 = self.att3(d3, e3)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        e2 = self.att2(d2, e2)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        e1 = self.att1(d1, e1)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        return self.out(d1)

class HybridLowLightEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.swinir = SwinIRDenoiser()
        self.unet = AttentionUNet()
    
    def forward(self, x):
        x = self.swinir(x) + x
        x = self.unet(x)
        return torch.clamp(x, 0, 1)

# ==================== FLASK APP SETUP ====================

app = Flask(__name__)
app.secret_key = "secret_key_for_flash_messages"

# 1. SETUP DEVICE & MODEL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Server starting. Using device: {device}")

model = HybridLowLightEnhancer()
model_path = 'lol-datasetmodel.pth' 

try:
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully.")
    else:
        print(f"⚠️ WARNING: Model file '{model_path}' not found. Using random weights.")
        model.to(device)
except Exception as e:
    print(f"❌ Error loading model: {e}")

# 2. IMAGE PREPROCESSING HELPERS
transform_model = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def run_inference(pil_img):
    """Core function to run model on a PIL image"""
    original_size = pil_img.size
    
    # Preprocess
    input_tensor = transform_model(pil_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Post-process
    output_tensor = output_tensor.squeeze(0).cpu()
    output_pil = transforms.ToPILImage()(output_tensor)
    
    # Resize back
    output_pil = output_pil.resize(original_size, Image.Resampling.BICUBIC)
    return output_pil

def process_image_bytes(image_bytes):
    """Helper for Image Upload Route"""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    enhanced_pil = run_inference(img)
    return img, enhanced_pil

def to_base64(pil_img):
    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=95)
    return base64.b64encode(img_io.getvalue()).decode('utf-8')

# 3. VIDEO GENERATOR
def gen_frames():
    """Generates frames from webcam, enhances them, and yields MJPEG stream"""
    camera = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 1. Convert OpenCV BGR to PIL RGB
        cv2_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_rgb)

        # 2. Run Inference (This is the slow part!)
        # Optimization tip: You might want to skip frames or resize smaller if laggy
        try:
            enhanced_pil = run_inference(pil_img)
        except Exception as e:
            print(f"Inference error: {e}")
            enhanced_pil = pil_img # Fallback to original

        # 3. Convert PIL Enhanced back to OpenCV BGR
        enhanced_np = np.array(enhanced_pil)
        enhanced_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)

        # 4. Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', enhanced_bgr)
        frame_bytes = buffer.tobytes()

        # 5. Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

# 4. ROUTES
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', mode='image')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
        
    if file:
        try:
            img_bytes = file.read()
            original_pil, enhanced_pil = process_image_bytes(img_bytes)
            
            original_b64 = to_base64(original_pil)
            enhanced_b64 = to_base64(enhanced_pil)
            
            return render_template('index.html', 
                                   mode='image',
                                   original_img=original_b64, 
                                   enhanced_img=enhanced_b64,
                                   filename=file.filename)
                                   
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_video')
def live_video():
    return render_template('index.html', mode='video')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)