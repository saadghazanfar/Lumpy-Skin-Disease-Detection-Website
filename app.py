"""
Flask Backend - LSDDetect
TFLite inference + Grad-CAM heatmap generation
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, base64, cv2

app = Flask(__name__)
CORS(app)

MODEL_PATH    = 'model1.tflite'
H5_MODEL_PATH = 'final_model.h5'   # needed only for real Grad-CAM

interpreter   = None
gradcam_model = None


# ── Custom layers (only needed when loading .h5 for Grad-CAM) ────────────────

@tf.keras.utils.register_keras_serializable()
class PSAMModule(tf.keras.layers.Layer):
    def __init__(self, filters, name="psam", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
    def build(self, input_shape):
        f = self.filters
        self.conv1x1       = tf.keras.layers.Conv2D(f//4, 1, padding='same', activation='relu')
        self.conv3x3_dw    = tf.keras.layers.DepthwiseConv2D(3, padding='same')
        self.conv3x3_pw    = tf.keras.layers.Conv2D(f//4, 1, activation='relu')
        self.conv5x5_dw    = tf.keras.layers.DepthwiseConv2D(5, padding='same')
        self.conv5x5_pw    = tf.keras.layers.Conv2D(f//4, 1, activation='relu')
        self.conv_dilated  = tf.keras.layers.Conv2D(f//4, 3, padding='same', dilation_rate=2, activation='relu')
        self.global_pool   = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.fusion_conv   = tf.keras.layers.Conv2D(f, 1, padding='same', activation='relu')
        self.bn            = tf.keras.layers.BatchNormalization()
        self.spatial_conv1 = tf.keras.layers.Conv2D(f//2, 3, padding='same', activation='relu')
        self.spatial_conv2 = tf.keras.layers.Conv2D(1, 1, padding='same', activation='sigmoid')
        super().build(input_shape)
    def call(self, inputs, training=None):
        p1 = self.conv1x1(inputs)
        p2 = self.conv3x3_pw(self.conv3x3_dw(inputs))
        p3 = self.conv5x5_pw(self.conv5x5_dw(inputs))
        p4 = self.conv_dilated(inputs)
        ms = tf.concat([p1, p2, p3, p4], axis=-1)
        h, w = tf.shape(inputs)[1], tf.shape(inputs)[2]
        up   = tf.image.resize(self.global_pool(ms), [h, w], method='bilinear')
        f2   = self.bn(self.fusion_conv(tf.concat([ms, up], axis=-1)), training=training)
        att  = self.spatial_conv2(self.spatial_conv1(f2))
        return inputs * att + f2
    def get_config(self):
        c = super().get_config(); c['filters'] = self.filters; return c


@tf.keras.utils.register_keras_serializable()
class ECAModule(tf.keras.layers.Layer):
    def __init__(self, gamma=2, b=1, name="eca", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma; self.b = b
    def build(self, input_shape):
        ch = input_shape[-1]
        t  = int(abs((np.log2(ch) + self.b) / self.gamma))
        self.kernel_size = max(t if t % 2 else t + 1, 3)
        self.conv1d = tf.keras.layers.Conv1D(1, self.kernel_size, padding='same', use_bias=False)
        super().build(input_shape)
    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        att = self.conv1d(tf.expand_dims(gap, -1))
        att = tf.nn.sigmoid(tf.squeeze(att, -1))
        att = tf.reshape(att, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * att
    def get_config(self):
        c = super().get_config(); c.update({'gamma':self.gamma,'b':self.b}); return c


# ── Model loading ─────────────────────────────────────────────────────────────

def load_tflite_model():
    global interpreter
    try:
        if os.path.exists(MODEL_PATH):
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            i = interpreter.get_input_details()
            o = interpreter.get_output_details()
            print(f"✓ TFLite loaded | input={i[0]['shape']} output={o[0]['shape']}")
        else:
            print(f"⚠  {MODEL_PATH} not found — mock-prediction mode active")
    except Exception as e:
        print(f"⚠  TFLite error: {e}")

def load_gradcam_model():
    global gradcam_model
    custom = {'PSAMModule': PSAMModule, 'ECAModule': ECAModule}
    if os.path.exists(H5_MODEL_PATH):
        try:
            gradcam_model = tf.keras.models.load_model(
                H5_MODEL_PATH, custom_objects=custom, compile=False)
            print(f"✓ Grad-CAM model loaded from {H5_MODEL_PATH}")
        except Exception as e:
            print(f"⚠  Grad-CAM model error: {e}")
    else:
        print(f"⚠  {H5_MODEL_PATH} not found — Grad-CAM will use approximation")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((300, 300))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)   # (1, 300, 300, 3)


# ── TFLite inference ──────────────────────────────────────────────────────────

def run_tflite(processed: np.ndarray) -> float:
    i = interpreter.get_input_details()
    o = interpreter.get_output_details()
    interpreter.set_tensor(i[0]['index'], processed.astype(np.float32))
    interpreter.invoke()
    return float(interpreter.get_tensor(o[0]['index'])[0][0])


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def ndarray_to_b64(arr: np.ndarray) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_gradcam(image_bytes: bytes) -> str:
    orig = np.array(
        Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((300, 300)))

    # Real Grad-CAM (requires full Keras model)
    if gradcam_model is not None:
        try:
            last_conv = None
            for layer in reversed(gradcam_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            if last_conv is None:
                return ndarray_to_b64(orig)

            grad_model = tf.keras.models.Model(
                inputs  = gradcam_model.inputs,
                outputs = [gradcam_model.get_layer(last_conv).output,
                           gradcam_model.output]
            )
            img_t = tf.cast(preprocess_image(image_bytes), tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(img_t)
                conv_out, preds = grad_model(img_t)
                loss = preds[:, 0]

            grads   = tape.gradient(loss, conv_out)
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam     = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
            cam     = np.maximum(cam, 0)
            if cam.max() > 0:
                cam /= cam.max()

            cam_r   = cv2.resize(cam, (300, 300))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
            heatmap = heatmap[:, :, ::-1]   # BGR -> RGB
            blended = np.clip(0.55 * orig + 0.45 * heatmap, 0, 255).astype(np.uint8)
            return ndarray_to_b64(blended)
        except Exception as e:
            print(f"Grad-CAM error: {e}")

    # Approximation: local gradient saliency (no .h5 needed)
    gray    = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    diff    = np.abs(gray - blurred)
    if diff.max() > 0:
        diff /= diff.max()
    diff    = cv2.GaussianBlur(diff, (15, 15), 0)
    if diff.max() > 0:
        diff /= diff.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * diff), cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1]
    blended = np.clip(0.6 * orig + 0.4 * heatmap, 0, 255).astype(np.uint8)
    return ndarray_to_b64(blended)


# ── Business logic ────────────────────────────────────────────────────────────

def compute_env_risk(env: dict) -> dict:
    t  = env.get('meanTemp', 0)
    h  = env.get('humidity', 0)
    tr = 'HIGH' if t > 25 else ('MEDIUM' if t > 20 else 'LOW')
    hr = 'HIGH' if h > 70 else ('MEDIUM' if h > 50 else 'LOW')
    return {
        'tempRisk':           tr,
        'humidityRisk':       hr,
        'climateSuitability': 'FAVORABLE'
                              if tr in ('HIGH','MEDIUM') or hr in ('HIGH','MEDIUM')
                              else 'UNFAVORABLE'
    }


def build_recommendations(is_lumpy: bool, er: dict) -> list:
    if is_lumpy:
        return [
            "IMMEDIATE ISOLATION: Separate the affected animal from the herd immediately",
            "VETERINARY CONSULTATION: Contact a veterinarian for professional diagnosis and treatment",
            "BIOSECURITY MEASURES: Implement strict hygiene protocols to prevent disease spread",
            "HERD MONITORING: Examine all animals daily for early detection of symptoms",
            "VACCINATION: Ensure all healthy animals are vaccinated per official guidelines",
            "DOCUMENTATION: Record case details for epidemiological tracking and insurance"
        ]
    recs = [
        "HEALTHY STATUS: Animal appears normal — maintain regular health monitoring",
        "PREVENTIVE CARE: Continue scheduled vaccinations and health checkups",
        "RECORD KEEPING: Document this assessment for future reference"
    ]
    if er['tempRisk'] in ('HIGH','MEDIUM') or er['humidityRisk'] in ('HIGH','MEDIUM'):
        recs += [
            "ENVIRONMENTAL ALERT: Current climate conditions favor disease transmission",
            "ENHANCED VIGILANCE: Increase monitoring frequency due to favorable conditions",
            "HYGIENE PROTOCOL: Ensure proper ventilation and sanitation in animal housing"
        ]
    return recs


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json', mimetype='application/manifest+json')

@app.route('/sw.js')
def service_worker():
    r = send_from_directory('.', 'sw.js', mimetype='application/javascript')
    r.headers['Service-Worker-Allowed'] = '/'
    r.headers['Cache-Control']          = 'no-cache'
    return r

@app.route('/icons/<path:filename>')
def serve_icon(filename):
    return send_from_directory('icons', filename)

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_bytes = request.files['image'].read()  # read ONCE, reuse for Grad-CAM

        env_keys = ['longitude','latitude','meanTemp','minTemp','maxTemp',
                    'precipitation','cloudCover','humidity',
                    'evapotranspiration','vaporPressure','wetDayFreq']
        env = {k: float(request.form.get(k, 0)) for k in env_keys}

        # Inference
        processed = preprocess_image(image_bytes)
        if interpreter is not None:
            raw = run_tflite(processed)
        else:
            tf_f = 0.55 if env['meanTemp'] > 25 else 0.25
            hf_f = 0.25 if env['humidity']  > 70 else 0.08
            raw  = min(0.92, tf_f + hf_f + np.random.random() * 0.15)

        is_lumpy   = raw > 0.5
        confidence = float(raw if is_lumpy else 1.0 - raw)
        er         = compute_env_risk(env)
        recs       = build_recommendations(is_lumpy, er)
        gcam_b64   = generate_gradcam(image_bytes)

        return jsonify({
            'success': True,
            'prediction': {
                'isLumpy':    bool(is_lumpy),
                'confidence': confidence,
                'rawScore':   float(raw),
                'status':     'LUMPY DETECTED' if is_lumpy else 'NORMAL'
            },
            'environmentalRisk': er,
            'recommendations':   recs,
            'gradcam':           gcam_b64,
            'envData':           env,
            'location': {
                'latitude':  env['latitude'],
                'longitude': env['longitude']
            }
        })
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':        'healthy',
        'model':         MODEL_PATH,
        'model_loaded':  interpreter is not None,
        'gradcam_ready': gradcam_model is not None
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  LSDDetect — TFLite Inference + Grad-CAM")
    print("=" * 60)
    load_tflite_model()
    load_gradcam_model()
    print(f"\n  http://localhost:5000\n")
    app.run(debug=True, host='0.0.0.0', port=5000)