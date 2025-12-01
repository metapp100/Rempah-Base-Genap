import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO # Asumsi model best.pt adalah model YOLO

# --- Konfigurasi ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'best.pt' # Pastikan file model ada di direktori yang sama

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder uploads ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Fungsi untuk memuat model ML (Dilakukan sekali saat server dimulai)
try:
    print(f"Memuat model dari: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model. Pastikan file {MODEL_PATH} ada. Detail: {e}")
    model = None # Set None jika gagal, agar aplikasi tidak crash

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routing ---

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        # Tampilkan halaman upload
        return render_template('index.html')

    if request.method == 'POST':
        # Periksa apakah ada file di request
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada bagian file di request'}), 400
        
        file = request.files['file']

        # Jika user tidak memilih file, browser juga mengirimkan file kosong
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # --- Bagian Deteksi Rempah Menggunakan Model ---
            if model is None:
                return jsonify({'error': 'Model deteksi belum siap atau gagal dimuat.'}), 500
            
            try:
                # 1. Jalankan inferensi
                # Model YOLO ultralytics akan mengembalikan hasil dalam objek Results
                results = model(filepath) 
                
                detections = []
                
                # 2. Proses hasil deteksi
                # Kita asumsikan model Anda mendeteksi objek
                for r in results:
                    # r.boxes.data berisi tensor [x1, y1, x2, y2, confidence, class]
                    for box in r.boxes.data:
                        confidence = round(box[4].item() * 100, 2)
                        class_id = int(box[5].item())
                        
                        # Ambil nama kelas/rempah dari model (pastikan model.names sudah ada)
                        spice_name = model.names.get(class_id, f"Kelas tidak dikenal ({class_id})")
                        
                        detections.append({
                            'name': spice_name,
                            'confidence': confidence
                        })

                # Hapus file setelah diproses
                os.remove(filepath)
                
                # Kembalikan hasil ke frontend
                return jsonify({'status': 'success', 'detections': detections})

            except Exception as e:
                print(f"Error saat inferensi: {e}")
                os.remove(filepath) # Tetap hapus file jika terjadi error
                return jsonify({'error': f'Gagal memproses gambar: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Tipe file tidak didukung'}), 400

if __name__ == '__main__':
    # Mode Debug = True hanya untuk pengembangan
    app.run(debug=True)