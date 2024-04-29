# Number Detection on Production Code 


**Number Detection API** digunakan untuk melakukan proses deteksi dan pengenalan baris angka yang tertera pada gambar sebagai kode produksi. Teknologi yang digunakan bernama  OCR (Optical Character Recognition) dari [EasyOCR](https://github.com/JaidedAI/EasyOCR/tree/master/easyocr) dan [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Sejauh ini, PaddleOCR memberikan result yang jauh lebih baik dibandingkan dengan EasyOCR.

### Quick Summary
Nama Project : **Number Detection on Production Code** <br>
Environment : **`paddleocr`** <br>
Algoritma Pendekatan : OCR (Optical Character Recognition) <br>
Current Model : **PaddleOCR from [PaddlePaddle](https://github.com/PaddlePaddle/PaddleOCR)** <br>


# <div align="left"><h3>API Documentation</h3></div>

# Base URL

```bash
http://ai.quick.com
```
Seluruh jenis request menuju server AI menggunakan base URL tersebut.

<br>

# Endpoints

- ##  Get Number Detection's API Info
    Endpoint ini digunakan untuk mendapatkan info bahwa servis API untuk Number Detection telah aktif. Metode yang digunakan adalah  **`GET`**.
    <br>

    - **Endpoint**
        ```bash
        GET   /number-detection-prodcode/predict
        ```

    <br>

    - **Successful Response**
        ```html
        Number detection production code
        ```
    <br>

- ##  Perform Number Detection and Recognition
    Endpoint ini digunakan untuk melakukan fungsi utama yaitu deteksi dan pengenalan angka dalam gambar input. Metode yang digunakan adalah **`POST`**.
    <br>

    - **Endpoint**
        ```bash
        POST   /number-detection-prodcode/predict
        ```
        **Request Body** `(form-data)` :
        * **`image`** _(file, required)_ : gambar produk sebagai input.

    <br>

    - **Example Request using CURL** <br>
        Contoh gambar:
        <div align="center"><img src="img/Images.jpg", width= 600px></div>

        ```bash
        curl --request POST 'http://ai.quick.com/number-detection-prodcode/predict' \
        --header 'Host: ai.quick.com' \
        --header 'Content-Type: multipart/form-data' \
        --form 'image=@"/path/To/yourFolder/Image.jpg"' \
        ```

    <br>

    - **Successful Response**
        ```json
        {
            "data": [
                {
                    "bbox_height": 361,
                    "bbox_width": 394,
                    "bbox_xcenter": 427,
                    "bbox_ycenter": 1321,
                    "pred": "2",
                    "xpred": 624,
                    "ypred": 1715
                },
                {
                    "bbox_height": 335,
                    "bbox_width": 289,
                    "bbox_xcenter": 789,
                    "bbox_ycenter": 1308,
                    "pred": "7",
                    "xpred": 934,
                    "ypred": 1597
                },
                {
                    "bbox_height": 335,
                    "bbox_width": 295,
                    "bbox_xcenter": 1071,
                    "bbox_ycenter": 1321,
                    "pred": "0",
                    "xpred": 1218,
                    "ypred": 1616
                },
                {
                    "bbox_height": 348,
                    "bbox_width": 309,
                    "bbox_xcenter": 1354,
                    "bbox_ycenter": 1328,
                    "pred": "9",
                    "xpred": 1508,
                    "ypred": 1637
                },
                {
                    "bbox_height": 355,
                    "bbox_width": 335,
                    "bbox_xcenter": 1650,
                    "bbox_ycenter": 1315,
                    "pred": "2",
                    "xpred": 1818,
                    "ypred": 1650
                },
                {
                    "bbox_height": 355,
                    "bbox_width": 269,
                    "bbox_xcenter": 1972,
                    "bbox_ycenter": 1334,
                    "pred": "3",
                    "xpred": 2106,
                    "ypred": 1603
                },
                {
                    "bbox_height": 348,
                    "bbox_width": 302,
                    "bbox_xcenter": 2452,
                    "bbox_ycenter": 1393,
                    "pred": "2",
                    "xpred": 2603,
                    "ypred": 1695
                },
                {
                    "bbox_height": 341,
                    "bbox_width": 289,
                    "bbox_xcenter": 2985,
                    "bbox_ycenter": 1453,
                    "pred": "0",
                    "xpred": 3130,
                    "ypred": 1742
                },
                {
                    "bbox_height": 322,
                    "bbox_width": 282,
                    "bbox_xcenter": 3261,
                    "bbox_ycenter": 1512,
                    "pred": "6",
                    "xpred": 3402,
                    "ypred": 1794
                },
                {
                    "bbox_height": 315,
                    "bbox_width": 263,
                    "bbox_xcenter": 3550,
                    "bbox_ycenter": 1558,
                    "pred": "5",
                    "xpred": 3682,
                    "ypred": 1821
                }
            ],
            "img_height": 3120,
            "img_width": 4208
        }
        ```
    <br>

# Error Handling
Indikasi error yang terjadi baik pada request maupun pada service API, ditunjukkan dengan respon berupa HTTP status code selain kode 200. Penanganan error berbeda-beda sesuai dengan jenis error yang ditemukan. Berikut beberapa contoh error message yang dimaksud:

* **400** _bad request_

* **403** _forbidden_

* **404** _notfound_

* **405** _method not allowed_

* **408** _request timeout_

* **500** _internal server error_

* **502** _bad gateway_

* **503** _service unavailable_

* **504** _gateway timeout_
<br>