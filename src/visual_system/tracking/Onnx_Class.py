import onnxruntime as ort
import numpy as np
import cv2
import time


# Класс для работы с ONNX моделью
class CnnOnnx:
    def __init__(self, weights_path, size=640, cuda=True):
        """
        Инициализация класса.

        :param weights_path: Путь к весам модели.
        :param size: Размер изображения для ввода в модель.
        :param cuda: Использовать ли GPU (CUDA).
        """
        self.size = size
        self.weights_path = weights_path
        # Выбор провайдеров для выполнения (CUDA для GPU или CPU)
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        print("Используемые провайдеры: ", self.providers)
        # Создание сессии для выполнения модели
        self.session = ort.InferenceSession(self.weights_path, providers=self.providers)
        # Получение имен входов и выходов модели
        self.outname = [i.name for i in self.session.get_outputs()]
        self.inname = [i.name for i in self.session.get_inputs()]

    def detect_image(self, image, conf_threshold=0.1, nms_threshold=0.8):
        """
        Обнаружение объектов на изображении.

        :param image: Исходное изображение.
        :param conf_threshold: Порог вероятности для обнаружения.
        :param nms_threshold: Порог для NMS (non-maximum suppression).
        :return: Коробки, вероятности и классы объектов.
        """
        # Конвертация изображения в RGB формат
        processed_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # Изменение размера изображения и добавление отступов
        processed_img, ratio, dwdh = letterbox(processed_img, auto=False, new_shape=(self.size, self.size))
        # Транспонирование осей изображения для соответствия формату ONNX
        processed_img = processed_img.transpose((2, 0, 1))
        # Добавление измерения для батча
        processed_img = np.expand_dims(processed_img, 0)
        # Конвертация массива для последовательности памяти
        processed_img = np.ascontiguousarray(processed_img)
        # Нормализация изображения
        im = processed_img.astype(np.float32)
        im /= 255

        # Подача изображения на вход модели и выполнение инференции
        inp = {self.inname[0]: im}
        start = time.time()
        outputs = self.session.run(self.outname, inp)[0]
        end = time.time()
        # Время выполнения инференции
        res = end - start
        h, w, _ = image.shape
        boxes = []
        scores = []
        classes = []
        # Обработка результатов инференции
        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            if cls_id == 0 and score > 0.6:
                box = np.array([x0, y0, x1, y1])
                box -= np.array(dwdh * 2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                boxes.append(box)
                scores.append(score)
                classes.append(cls_id)

        # Применение NMS для подавления слабых прогнозов
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
            boxes = [boxes[i] for i in indices.ravel()]
            scores = [scores[i] for i in indices.ravel()]
            classes = [classes[i] for i in indices.ravel()]

        return boxes, scores, classes

    def draw_boxes(self, boxes, image):
        """
        Рисование обнаруженных объектов на изображении.

        :param boxes: Коробки обнаруженных объектов.
        :param image: Исходное изображение.
        :param classes: Классы обнаруженных объектов.
        :return: Изображение с нарисованными объектами.
        """

        h, w, _ = image.shape

        # Отображение обнаруженных объектов на изображении
        for box in boxes:
            color = (128, 128, 0)
            cv2.rectangle(image, box[:2], box[2:], color, 2)
            cv2.putText(image, "Person", (box[0], box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,
                        thickness=2)
        return image


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    """
    Изменение размера и добавление отступов к изображению.

    :param im: Исходное изображение.
    :param new_shape: Желаемый размер изображения.
    :param color: Цвет отступов.
    :param auto: Автоматический режим отступов.
    :param scaleup: Увеличение размера изображения.
    :param stride: Шаг для отступов.
    :return: Изображение с измененным размером и отступами.
    """
    # Текущий размер изображения
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Соотношение масштаба (новый / старый)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Вычисление отступов
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    # Изменение размера изображения
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # Добавление отступов к изображению
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)
