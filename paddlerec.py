import cv2
import string
import Levenshtein
from PIL import Image
from paddleocr import PaddleOCR

# ocr = PaddleOCR(lang='en')

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

# Fungsi untuk mencari karakter yang paling mendekati karakter diizinkan
def find_best_character(target_char, candidates):
    min_distance = float('inf')
    best_char = None
    for candidate in candidates:
        distance = Levenshtein.distance(target_char, candidate)
        if distance < min_distance:
            min_distance = distance
            best_char = candidate
    return best_char

