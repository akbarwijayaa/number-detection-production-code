# Project Summary
Project Name : **Number Detection Production Code**
<br>
Environment : **`craft-pytorch --> include library`**
<br>
Algorithm : <ol> 
                <li>**[CRAFT-Pytorch](https://github.com/clovaai/CRAFT-pytorch) from clovaai**</li>
                <li>**[paddleocr](https://github.com/PaddlePaddle/PaddleOCR) from PaddlePaddle**</li>
            </ol>
<br>
Current Model : **craft_mlt_25k.pth + craft_refiner_CTW1500.pth**
<br>
Folder for Development : /home/serverai/Project/number_detection/

<br>

# Quickstart
Clone repository ini, kemudian install seluruh dependensi yang terdapat dalam `requirements.txt`.
```bash
$ git clone http://gitlab.quick.com/artificial-intelligence/number-detection-production-code.git     #clone
$ cd number-detection-production-code
$ conda activate craft-pytorch
```
<br>

# Dataset
Untuk dataset kami menggunakan pretrained model dari craft-pytorch dan paddleocr.
<br>

# Training and Testing
Karena kami menggunakan pretrained model dari library, maka tidak ada proses testing. Namun kami melakukan beberapa pengujian menggunakan beberapa model.

## Parameter Tunning
Pada source code pipeline yang mana digunakan untuk mencari text yang terdapat didalam gambar terdapat function detection yang mana argumen yang berada didalamnya bisa ditunning untuk menambah akurasi model.
``` python
def detection(image_bytes, use_cuda = False, use_refine = True, use_poly = False,
              text_threshold = 0.9, link_threshold = 0.4, low_text = 0.4, 
              canvas_size = 1280, mag_ratio = 1.5, show_time = False):
    ...
```

<br>

# Source Code Documentation
## > Main API Program 
**File Location : `app.py`** <br> Aplikasi Flask python untuk membuat REST API yang berkaitan dengan Number Detection. Rest API ini memiliki satu endpoint utama, yaitu /`number-detection-prodcode/predict`.
- #### endpoint `/number-detection-prodcode/predict`
    Endpoint ini menjalankan rangkaian proses deteksi angka yang terdapat dalam gambar input. Proses tersebut dideklarasikan dalam sebuah fungsi `predict()`.

    ```python
    @app.route(DETECTION_URL, methods=["POST", "GET"])
    def predict():
        if not request.method == "POST":
            return "<h1>Number detection production code</h1>"

        image_file = request.files.getlist("image")
        for img in image_file:
            image_bytes = img.read()
            # convert to numpy.ndarray
            img_b64 = base64.b64encode(image_bytes)
            nparr = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            img_ndarray = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # get shape
            img_width, img_height = get_shape(img_ndarray)
            img_preprocc = applyHistEqualized(img_ndarray)
            # process
            data_csv, img_arr = detection(img_preprocc)
            crop_img = imageCrop(data_csv, img_arr)
            result_images, bbox = crop_img.get_images()
            recog_result = recognize(result_images)
            # recog_result = recognition(result_images)
            df = mkdf(bbox, recog_result)
            dict_df = df.to_dict(orient='records')
            result = {
                'data':dict_df,
                'img_width':img_width,
                'img_height':img_height
            }
            return result
    ```

<br>

- **`mkdf()` dan `get_shape()`**<br>
    Fungsi `mkdf()` mengambil informasi bounding box dari hasil deteksi untuk disusun ke dalam sebuah DataFrame. Kemudian, ditambahkan kolom ['xpred', 'ypred', 'pred'] pada DataFrame tersebut. <br>
    Terakhir yaitu `get_shape()` yang mengambil ukuran dari gambar input.

    ```python
    def mkdf(bbox, recog_result):
        df = pd.DataFrame(bbox, columns=['bbox_xcenter', 'bbox_ycenter', 'bbox_width', 'bbox_height'])
        df['xpred'] = round(df['bbox_xcenter']+(df['bbox_width']/2)).astype('int')
        df['ypred'] = round(df['bbox_ycenter']+df['bbox_width']).astype('int')
        df['pred'] = recog_result
        return df

    def get_shape(img_ndarray):
        im_sz = img_ndarray.shape
        return im_sz[1], im_sz[0]
    ```

<br>

## > Image Processing using `applyHistEqualized()`
**File Location : `preprocc.py`** <br> Module ini berfungsi untuk melakukan *image processing* dengan *Hisogram Equalization* OpenCV2. Histogram Equalization sendiri merupakan proses meratakan distribusi intensitas keabuan pixel dalam gambar, dengan tujuan meningkatkan kontras.

```python
def applyHistEqualized(img_ndarray):
    gray = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    togray = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return togray
```

<br>

## > Numbers Bounding Box Detection
**File Location : `pipeline.py`** <br> Program ini mendeteksi angka pada gambar menggunakan model CRAFT (Character Region Awareness for Text Detection). 

- **`copyStateDict()`** : Mengonversi kunci-kunci dalam kamus (state dictionary) suatu model. Ini berguna untuk menangani perbedaan dalam penyimpanan kunci-kunci model.
    ```python
    def copyStateDict(state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict
    ```

- **`detection()`**: Menjalankan proses deteksi bounding box dengan CRAFT dan RefineNET.

    ```python
    def detection(image_bytes, use_cuda = False, use_refine = True, use_poly = False,
                text_threshold = 0.9, link_threshold = 0.4, low_text = 0.4, 
                canvas_size = 1280, mag_ratio = 1.5, show_time = False):
        
        base_path = dirname(abspath(__file__))
        model_path = join(base_path, 'models')
        result_path = join(base_path, 'result')
        trained_model_path = join(model_path, 'craft_mlt_25k.pth')
        refine_path = join(model_path, 'craft_refiner_CTW1500.pth')
            
        data=pd.DataFrame(columns=['test','word_bboxes', 'pred_words', 'align_text'])
        data['test'] = ['test']
        
        net = CRAFT()     # initialize

        if use_cuda:
            net.load_state_dict(test.copyStateDict(torch.load(trained_model_path)))
        else:
            net.load_state_dict(test.copyStateDict(torch.load(trained_model_path, map_location='cpu')))

        if use_cuda:
            net = net.cuda()
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = False

        net.eval()

        # menggunakan Refiner Net
        refine_net = None
        if use_refine:
            from craftocr.refinenet import RefineNet
            refine_net = RefineNet()
            if use_cuda:
                refine_net.load_state_dict(copyStateDict(torch.load(refine_path)))
                refine_net = refine_net.cuda()
                refine_net = torch.nn.DataParallel(refine_net)
            else:
                refine_net.load_state_dict(copyStateDict(torch.load(refine_path, map_location='cpu')))

            refine_net.eval()
            use_poly = True

        t = time.time()

        # load image data
        image = imgproc.loadImageFromNdArray(image_bytes)
        bboxes, polys, score_text, det_scores = test.test_net(net, image, text_threshold, link_threshold, low_text, use_cuda, use_poly, canvas_size, mag_ratio, show_time, refine_net)
        bbox_score={}

        for box_num in range(len(bboxes)):
            key = str (det_scores[box_num])
            item = bboxes[box_num]
            bbox_score[key]=item
        
        # menyimpan ke dalam DataFrame
        data['word_bboxes'][0]=bbox_score
        
        return data, image
    ```

<br>

## > Crop Image Based on Its Bounding Box
**File Location : `crop_image_words.py`** <br> 
Module ini memuat proses lanjutan dari `pipeline.py`. Setelah didapatkan bounding box angka-angka dalam gambar, kemudian dilakukan cropping dan diolah untuk diambil informasi hasil deteksinya secara terstruktur. 

- `__init__` :  Pada tahap inisialisasi, beberapa variabel path disiapkan untuk menyimpan data dan model yang dibutuhkan.

    ```python 
    def __init__(self, df_temp, img_nparr):
        self.df_temp = df_temp
        self.image = img_nparr
        self.score_bbox = str(self.df_temp['word_bboxes'][0]).split('),')
        self.bbox_coordinate = self.generate_words(self.score_bbox)
        self.sorted_bboxes = sorted(self.bbox_coordinate, key=lambda bbox: bbox[0])
    ```

- `generate_words()` : Fungsi ini melakukan pengolahan data bounding box dan menghasilkan sebuah daftar (bbox) yang berisi informasi terstruktur tentang bounding box untuk setiap kata yang terdeteksi.

    ```python
    def generate_words(self, score_bbox):
        num_bboxes = len(score_bbox)
        bbox = []
        for num in range(num_bboxes):
            bbox_coords = score_bbox[num].split(':')[-1].split(',\n')
            if bbox_coords!=['{}']:
                l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0]) # x1
                t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1]) # y1
                r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0]) # param W / x2
                t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1]) 
                r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
                b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
                l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
                b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']')) # param h
                pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
                
                # Parameter x y w h
                x = int(l_t)
                y = int(t_l)
                w = int(r_t - l_t)
                h = int(b_l - t_l)

                aaa = [x, y, w, h]
                bbox.append(aaa)
        return bbox
    ```

- `get_images()` : Fungsi ini melakukan cropping pada bagian-bagian gambar yang sesuai dengan koordinat bounding box yang telah diurutkan sebelumnya dan disimpan dalam `self.sorted_bboxes`.

    ```python
    def get_images(self):
        num_bboxes = len(self.score_bbox)
        a = 1
        result_images = []
        bbox_coor = []
        for x,y,w,h in self.sorted_bboxes:
            list_a = [x,y,w,h]
            for num in range(num_bboxes):
                bbox_coords = self.score_bbox[num].split(':')[-1].split(',\n')
                if bbox_coords!=['{}']:
                    l_t = float(bbox_coords[0].strip(' array([').strip(']').split(',')[0]) # x1
                    t_l = float(bbox_coords[0].strip(' array([').strip(']').split(',')[1]) # y1
                    r_t = float(bbox_coords[1].strip(' [').strip(']').split(',')[0]) # param W / x2
                    t_r = float(bbox_coords[1].strip(' [').strip(']').split(',')[1]) 
                    r_b = float(bbox_coords[2].strip(' [').strip(']').split(',')[0])
                    b_r = float(bbox_coords[2].strip(' [').strip(']').split(',')[1])
                    l_b = float(bbox_coords[3].strip(' [').strip(']').split(',')[0])
                    b_l = float(bbox_coords[3].strip(' [').strip(']').split(',')[1].strip(']')) # param h
                    pts = np.array([[int(l_t), int(t_l)], [int(r_t) ,int(t_r)], [int(r_b) , int(b_r)], [int(l_b), int(b_l)]])
                x1 = int(l_t)
                y1 = int(t_l)
                w1 = int(r_t - l_t)
                h1 = int(b_l - t_l)
                list_b = [x1,y1,w1,h1]
                word = self.crop(pts, self.image)       
                
                if list_a == list_b:
                    try:
                        result_images.append(word)
                        bbox_coor.append(list_b)
                        a = a+1
                    except:
                        continue 
        return result_images, bbox_coor
    ```

<br>

## > Recognition using Paddle OCR
**File Location : `paddlerec.py`** <br> Modul ini bertanggung jawab untuk menjalankan number recognition (OCR) dengan menggunakan PaddleOCR.

- `recognize()` : Fungsi ini dirancang untuk melakukan pengenalan teks pada gambar-gambar menggunakan PaddleOCR dan memproses hasilnya untuk mendapatkan teks yang sesuai dengan karakter yang diperbolehkan.

    ```python
    def recognize(np_img):
        allowed_chars = string.digits
        optimized_results = []
        ocr = PaddleOCR(lang='en')
        for image in np_img:
            # pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            result = ocr.ocr(img=image, det=False, rec=True, cls=True) # Main recognize
            # Memproses hasil OCR
            for line in result:
                for word_info in line:
                    print(word_info[0])
                    recognized_text = word_info[0]
                    if recognized_text and recognized_text[0] in allowed_chars:
                        optimized_text = recognized_text[0]  # Mengambil karakter pertama
                    else:
                        optimized_text = find_best_character(recognized_text, allowed_chars)
                    optimized_results.append(optimized_text)
        return optimized_results
    ```

- `find_best_character()`: Fungsi ini berguna untuk mencari karakter terbaik yang cocok dengan karakter target dari daftar kandidat yang diberikan, berdasarkan perhitungan jarak *Levenshtein*.

    ```python
    def find_best_character(target_char, candidates):
        min_distance = float('inf')
        best_char = None
        for candidate in candidates:
            distance = Levenshtein.distance(target_char, candidate)
            if distance < min_distance:
                min_distance = distance
                best_char = candidate
        return best_char
    ```
    
<br>

# Testing Program
Selalu lakukan testing program langsung menggunakans serverai. Lakukan ssh ke server ai dengan `serverai@192.168.168.195`. Gunakan environment yang sesuai dengan penjelasan diatas. Running program python seperti biasa, `python app.py`. Pastikan saat itu port tidak terpakai oleh aplikasi lain. Jika program sudah berjalan, lakukan pengujian dengan mengirimkan gambar sample dalam api.

_Lihat dokumentasi api selengkapnya [disini](http://ai.quick.com/documentation/number-detection-prodcode/)_
