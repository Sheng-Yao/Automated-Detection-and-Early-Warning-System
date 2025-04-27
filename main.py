
import time
# Logging module
import logging
# Image capturing and annotation
import cv2
from picamera2 import Picamera2
# object detection modules
import torch
import pathlib
import distutils
# Audio capturing
import pyaudio
# Audio detection modules
import wave
import librosa
import numpy as np
import tensorflow as tf
from skimage.transform import resize
# API access
import requests
import json
# LoRa module
from LoRaRF import SX127x, LoRaSpi, LoRaGpio
# File control (Move between pending and completed folder)
import os
import shutil
# GPS communication (serial)
import serial
import math
# PIR sensor
from gpiozero import MotionSensor
# Battery control
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

# Configure logging
logging.basicConfig(
    filename='system.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a',
    level=logging.INFO
)
logger = logging.getLogger("System")


CONFIG = {
    "camera": {
        "resolution": (640,640),
        "format": 'RGB888'
    },
    "image": {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "font_color": (255,255,255),
        "font_scale": 0.6,
        "font_thickness": 2,
        "text_background_height": 35,
        "box_color": (254,204,2),
        "time_skipping_duration": 6,
        "bulk_frame_count": 3,
        "elephant_confidence_threshold": 15,
        "human_confidence_threshold": 50
    },
    "audio": {
        "format": pyaudio.paInt16,
        "channel": 1,
        "sampling_rate": 44100,
        "chunk": 4096,
        "audio_duration": 5,
        "device_index": 0,
        "target_shape": (128,128),
        "elephant_confidence_threshold": 49,
        "gunshot_confidence_threshold": 40
    },
    "Files": {
        "human_path": "./Resources/completed/images/human/Image_",
        "elephant_image_pending_path": "./Resources/pending/images/elephant/Image_",
        "elephant_image_completed_path": "./Resources/completed/images/elephant/Image_",
        "elephant_audio_pending_path": "./Resources/pending/audios/elephant/Audio_",
        "elephant_audio_completed_path": "./Resources/completed/audios/elephant/Audio_",
    },
    "API": {
        "link": "https://api.wildtechalert.com/",
        "device_name": "wta-a03",
        "api_key": "d7970e19-38ac-41aa-b1f4-6141fcf89b67",
        "handshake_duration": 900
    },
    "LoRa":{
        "frequency": 433000000,
        "preamble_length": 6,
        "spreading_factor": 7,
        "bandwidth": 500000,
        "coding_rate": 5,
        "sync_word": 0x14
    }
}

###################################################################   Class   ###################################################################

class LoRa:
    def __init__(self):
        self.lora = SX127x(LoRaSpi(0,0),LoRaGpio(0,23),LoRaGpio(0,22))
        self.setup()

    def setup(self):
        self.lora.begin()
        self.lora.setFrequency(CONFIG['LoRa']['frequency'])
        self.lora.setPreambleLength(CONFIG['LoRa']['preamble_length'])
        self.lora.setCrcEnable(True)
        self.lora.setSpreadingFactor(CONFIG['LoRa']['spreading_factor']) # 12
        self.lora.setBandwidth(CONFIG['LoRa']['bandwidth']) # 125000
        self.lora.setCodeRate(CONFIG['LoRa']['coding_rate']) # 8
        self.lora.setSyncWord(CONFIG['LoRa']['sync_word'])
  
    def sendLoRaMessage(self,input):
        self.lora.beginPacket()
        self.lora.write(list(input.encode('ascii')), len(input))
        self.lora.endPacket()
        self.lora.wait()

class Image:
    def __init__(self):
        self.camera_setup()
        self.image_model_setup()
        self.annotation_setup()
        self.file_path_setup()

    def camera_setup(self):
        self.camera = Picamera2()
        self.resolution = CONFIG['camera']['resolution']
        self.format = CONFIG['camera']['format']
        self.camera.configure(self.camera.create_preview_configuration(main={"format": self.format,"size":self.resolution}))
        self.camera.start()
        self.time_skipping = CONFIG['image']['time_skipping_duration']
        self.bulk_frame_count = CONFIG['image']['bulk_frame_count']
        self.sharpen_kernel = np.array([
             [0, -1, 0],
             [-1, 5, -1],
             [0, -1, 0]
        ])

    def image_model_setup(self):
        self.image_model = torch.hub.load(
            "/home/user/Desktop/myvenv/yolov5",
            "custom",
            path="/home/user/Desktop/myvenv/best.pt",
            source="local",  # This ensures it's loading from a local file
            force_reload=True  # Clears cache in case of corruption
        )

    def annotation_setup(self):
        self.font = CONFIG['image']['font']
        self.font_scale = CONFIG['image']['font_scale']
        self.font_color = CONFIG['image']['font_color']
        self.font_thickness = CONFIG['image']['font_thickness']
        self.box_color = CONFIG['image']['box_color']
        self.text_background_height = CONFIG['image']['text_background_height']

    def file_path_setup(self):
        self.human_path = CONFIG["Files"]["human_path"]
        self.elephant_image_pending_path = CONFIG["Files"]["elephant_image_pending_path"]
        self.elephant_image_completed_path = CONFIG["Files"]["elephant_image_completed_path"]

    def imageFlow(self, timer):
        if time.time() - timer > self.time_skipping:
            highest_confidence_elephant = 0
            temp_frame_elephant = []
            temp_results_elephant = list()
            date_time_elephant = ''

            highest_confidence_human = 0
            temp_frame_human = []
            temp_results_human = list()
            date_time_human = ''

            for frame in range(self.bulk_frame_count):
                image_frame, results = self.imageDetection()
                labels, cordinates = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
                if len(labels) == 0:
                    continue
                else:

                    if 0 in labels:
                        temp_frame_elephant, temp_results_elephant, highest_confidence_elephant, date_time_elephant = self.getHighestConfidence(0, image_frame, results, labels, cordinates, highest_confidence_elephant, date_time_elephant, temp_frame_elephant, temp_results_elephant)
                    
                    if 1 in labels:
                        temp_frame_human, temp_results_human, highest_confidence_human, date_time_human = self.getHighestConfidence(1, image_frame, results, labels, cordinates, highest_confidence_human, date_time_human, temp_frame_human, temp_results_human)

            if highest_confidence_human != 0:
                temp_frame_human,_ = self.image_annotation(temp_frame_human, temp_results_human, date_time_human, 1)
                human_file_path = self.human_path + date_time_human + '.jpg'
                logger.info('Human detected | Confidence: ' + str(highest_confidence_human) + '%')
                print('Human detected | Confidence: ' + str(highest_confidence_human) + '%')

                if cv2.imwrite(human_file_path,temp_frame_human):
                    logger.info('Human image saved: ' + str(human_file_path))
                    print('Human image saved: ' + str(human_file_path))
                else:
                    logger.error('Failed to save image: ' + str(human_file_path))
                    print('Failed to save image: ' + str(human_file_path))
            
            if highest_confidence_elephant != 0:
                temp_frame_elephant,_ = self.image_annotation(temp_frame_elephant, temp_results_elephant, date_time_elephant, 0)
                elephant_file_path = self.elephant_image_pending_path + date_time_elephant + '.jpg'
                logger.info('Elephant detected | Confidence: ' + str(highest_confidence_elephant) + '%')
                print('Elephant detected | Confidence: ' + str(highest_confidence_elephant) + '%')
                
                if cv2.imwrite(elephant_file_path,temp_frame_elephant):
                    logger.info('Elephant image saved: ' + str(elephant_file_path))
                    print('Elephant image saved: ' + str(elephant_file_path))
                else:
                    logger.error('Failed to save image: ' + str(elephant_file_path))
                    print('Failed to save image: ' + str(elephant_file_path))

            timer = self.updateTimer()
            if highest_confidence_elephant == 0:
               return timer, '', 0
            else: 
                return timer, date_time_elephant, highest_confidence_elephant
        else:
            return timer, '', 0

    def updateTimer(self):
        return time.time()

    def getHighestConfidence(self, object_code, image_frame, results, labels, cordinates, current_highest_confidence, date_time, previous_frame, previous_results):
        for object in range(len(labels)):
            if labels[object - 1] == object_code:
                image_confidence = int(cordinates[object][4]*100)
                if image_confidence > current_highest_confidence:
                    date_time = time.strftime("%Y_%m_%d_%H_%M_%S")
                    current_highest_confidence = image_confidence
                    return image_frame, results, current_highest_confidence, date_time
        return previous_frame, previous_results, current_highest_confidence, date_time             

    def imageDetection(self):
        image_frame = self.camera.capture_array()
        image_frame = cv2.rotate(image_frame,cv2.ROTATE_90_CLOCKWISE)
        image_frame = cv2.filter2D(image_frame, -1, self.sharpen_kernel)
        results = self.image_model(image_frame)
        return image_frame, results

    # Frame annotation
    def image_annotation(self, image_frame, results, date_time, mode):
        date_time = time.strptime(date_time, "%Y_%m_%d_%H_%M_%S")
        date_time = time.strftime("%Y-%m-%d %H:%M:%S",date_time)

        # Annotation for date and time
        image_frame = cv2.putText(image_frame, date_time, [410,625], self.font, self.font_scale, self.font_color, self.font_thickness)
        
        if results != list():
            # Obtain the object information from the model result
            labels, cordinates = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
            objects = len(labels)
        else:
            objects = 0

        # Total object label
        image_frame = cv2.putText(image_frame, f"Total Targets: {objects}", [15, 30], self.font, self.font_scale, self.font_color, self.font_thickness)

        if objects > 0:
            # Get frame size
            x_shape, y_shape = image_frame.shape[1], image_frame.shape[0]

            # Highest confidence in the image
            highest_confidence = 0

            # Plot all detected object
            for object in range(objects):

                row = cordinates[object]

                # Capture results of each object
                x1, y1, x2, y2, confidence = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape), int(row[4]*100)

                isTargetObject = False

                if mode == 0:
                    # Elephant class detected
                    if labels[object - 1] == 0:
                        isTargetObject = True
                        text_width = 145 if confidence < 100 else 155
                        highest_confidence = max(highest_confidence, confidence)
                        object_name = ' Elephant   '
                        # Draw rectangle for detected object
                        cv2.rectangle(image_frame, (x1, y1), (x2, y2), self.box_color, self.font_thickness)
                        
                elif mode == 1:
                    # Human class detected
                    if labels[object - 1] == 1:
                        isTargetObject = True
                        text_width = 120 if confidence < 100 else 130
                        highest_confidence = max(highest_confidence, confidence)
                        object_name = ' Human  '
                        cv2.rectangle(image_frame, (x1, y1), (x2, y2), self.box_color, self.font_thickness)

                if isTargetObject:
                    # Make sure text doesn't exceed right border
                    text_width = min(text_width, x_shape - x1 - 1)

                    # Ideal top position of text is above box
                    text_y_top = y1 - self.text_background_height

                    # If that would go out of image, shift it as high as possible inside image
                    if text_y_top < 0:
                        text_y_top = 0

                    # Draw background rectangle
                    cv2.rectangle(
                        image_frame,
                        (x1, text_y_top),
                        (x1 + text_width, text_y_top + self.text_background_height),
                        self.box_color,
                        -1
                    )

                    # Draw text
                    text_baseline = text_y_top + int(self.text_background_height * 0.75)
                    cv2.putText(
                        image_frame,
                        object_name + f'{confidence}',
                        (x1, text_baseline),
                        self.font,
                        self.font_scale,
                        self.font_color,
                        self.font_thickness
                    )

        return image_frame, highest_confidence
    
    def livePreview(self):
        image_frame = self.camera.capture_array()
        cv2.imshow("Camera", image_frame)

class Audio:
    def __init__(self):
        self.microphone_setup()
        self.audio_model_setup()
        self.file_path_setup()

    def microphone_setup(self):
        self.microphone = pyaudio.PyAudio()
        self.format = CONFIG['audio']['format']
        self.channel = CONFIG['audio']['channel']
        self.sampling_rate = CONFIG['audio']['sampling_rate']
        self.chunk = CONFIG['audio']['chunk']
        self.audio_duration = CONFIG['audio']['audio_duration']
        self.device_index = CONFIG['audio']['device_index']
        self.target_shape = CONFIG['audio']['target_shape']

    def audio_model_setup(self):
        ELEPHANT_MODEL_PATH = "ele-or-not-ele.h5"
        GUNSHOT_MODEL_PATH = "gun-or-not-gun.h5"
        self.elephant_audio_model = tf.keras.models.load_model(ELEPHANT_MODEL_PATH)
        self.gunshot_audio_model = tf.keras.models.load_model(GUNSHOT_MODEL_PATH)
        self.elephant_confidence_threshold = CONFIG['audio']['elephant_confidence_threshold']
        self.gunshot_confidence_threshold = CONFIG['audio']['gunshot_confidence_threshold']

    def file_path_setup(self):
        self.elephant_audio_pending_path = CONFIG["Files"]["elephant_audio_pending_path"]
        self.elephant_audio_completed_path = CONFIG["Files"]["elephant_audio_completed_path"]

    def startRecording(self):
        logger.info('Audio recording started | Duration: ' + str(self.audio_duration))
        print('Audio recording started | Duration: ' + str(self.audio_duration))

        stream = self.microphone.open(format = self.format, rate = self.sampling_rate, \
                                        channels = self.channel, input_device_index = self.device_index, \
                                        input = True, frames_per_buffer=self.chunk)
        stream.start_stream()
        return stream

    def audioRecording(self, stream):
        sound_frame = []
        for ii in range(0, int((self.sampling_rate/self.chunk) * self.audio_duration)):
            data = stream.read(self.chunk,False)
            sound_frame.append(data)
        return sound_frame

    def stopRecording(self, stream):
        stream.stop_stream()
        stream.close()

    def audioPlayback(self, audio_path):
        play_wavefile = wave.open(audio_path,'rb')

        stream = self.microphone.open(format = self.microphone.get_format_from_width(play_wavefile.getsampwidth()),
                        channels = play_wavefile.getnchannels(),
                        rate = play_wavefile.getframerate(),
                        output = True)

        data = play_wavefile.readframes(self.chunk)

        while data:
            stream.write(data)
            data=play_wavefile.readframes(self.chunk)

        play_wavefile.close()
        stream.close()

    def saveWavFile(self, date_time, sound_frame):
        audio_file_path = self.elephant_audio_pending_path + date_time + '.wav'
        wavefile = wave.open(audio_file_path,'wb')
        wavefile.setnchannels(self.channel)
        wavefile.setsampwidth(self.microphone.get_sample_size(self.format))
        wavefile.setframerate(self.sampling_rate)
        wavefile.writeframes(b''.join(sound_frame))
        wavefile.close()
        return audio_file_path

    def convertToSpectrogram(self, audio_file_path):
        try:
            audio_data, sample_rate = librosa.load(audio_file_path, sr=22050, duration=10)

            if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 1e-6:
                logger.error('Silent or empty audio detected')
                print('Silent or empty audio detected')
                return None

            # Compute Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)

            # Normalize like in training
            max_val = np.max(mel_spectrogram)
            if max_val > 0:
                mel_spectrogram = mel_spectrogram / max_val

            # Resize to match model input using skimage.transform.resize
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), self.target_shape)
            # Reshape for model compatibility
            mel_spectrogram = np.reshape(mel_spectrogram, (1,) + self.target_shape + (1,))

            logger.info('Spectrogram generated | Dimension: ' + str(self.target_shape))

            return mel_spectrogram
        
        except Exception as e:
            logger.error(f"ERROR: Failed to process audio - {e}")
            print(f"ERROR: Failed to process audio - {e}")
            return None

    def audioDetection(self, mel_spectrogram):
        if mel_spectrogram is None:
            return
        
        print("Running Detection...")
        elephant_prediction = self.elephant_audio_model.predict(mel_spectrogram)[0][0] * 100
        gunshot_prediction = self.gunshot_audio_model.predict(mel_spectrogram)[0][0] * 100
        
        print(f"Elephant Confidence: {elephant_prediction:.2f}%")
        print(f"Gunshot Confidence: {gunshot_prediction:.2f}%")

        # Classification Logic
        if elephant_prediction < self.elephant_confidence_threshold and gunshot_prediction < self.gunshot_confidence_threshold:
            logger.info('No detection')
            print("Classified as: Not Elephant and Not Gunshot")
        elif elephant_prediction > self.elephant_confidence_threshold and gunshot_prediction > self.gunshot_confidence_threshold:
            logger.info('Elephant detected | Confidence: ' + str(elephant_prediction) + '%')
            print(f"Classified as: Elephant with confidence {elephant_prediction:.2f}%")
            logger.info('Gunshot detected | Confidence: ' + str(gunshot_prediction) + '%')
            print(f"Classified as: Gunshot with confidence {gunshot_prediction:.2f}%")
        elif elephant_prediction > self.elephant_confidence_threshold:
            logger.info('Elephant detected | Confidence: ' + str(elephant_prediction) + '%')
            print(f"Classified as: Elephant with confidence {elephant_prediction:.2f}%")
        else:
            logger.info('Gunshot detected | Confidence: ' + str(gunshot_prediction) + '%')
            print(f"Classified as: Gunshot with confidence {gunshot_prediction:.2f}%")

        return elephant_prediction

class API:
    def __init__(self):
        self.link = CONFIG['API']['link']
        self.device_name = CONFIG['API']['device_name']
        self.api_key = CONFIG['API']['api_key']
        self.pending_image_path = CONFIG['Files']['elephant_image_pending_path']
        self.pending_audio_path = CONFIG['Files']['elephant_audio_pending_path']
        self.handshake_timer = 0
        self.handshake_duration = CONFIG['API']['handshake_duration']

    def handshake(self):
        link = self.link + 'handshake'
        headers = {
            'x-device-name': self.device_name,
            'x-api-key': self.api_key
        }
        return requests.post(link, headers=headers)

    def detect(self, image_confidence, audio_confidence, date_time):

        audio_path = self.pending_audio_path + date_time + '.wav'
        image_path = self.pending_image_path + date_time + '.jpg'

        output_audio_path = 'Audio_' + date_time + '.wav'
        output_image_path = 'Image_' + date_time + '.jpg'

        date_time = time.strptime(date_time, "%Y_%m_%d_%H_%M_%S")
        date_time = time.strftime("%Y-%m-%dT%H:%M:%S",date_time)

        link = self.link + 'detect'
        headers = {'x-api-key': self.api_key}

        if audio_confidence == None:
            audio_confidence = 0
        
        if image_confidence == None:
            image_confidence = 0

        if audio_confidence > CONFIG['audio']['elephant_confidence_threshold']:
            payload = {
                'confidence_level_audio': round(audio_confidence / 100.0, 2),
                'confidence_level_camera': round(image_confidence / 100.0, 2),
                'device_name': self.device_name,
                'detected_at': date_time,
                'audio_detected': True,
                'camera_detected': True
            }
            files = {
                'sound_file': (output_audio_path,open(audio_path,'rb'),'audio/mpeg'),
                'image_file': (output_image_path,open(image_path,'rb'),'image/jpeg')
            }
        else:
            payload = {
                'confidence_level_audio': 0,
                'confidence_level_camera': round(image_confidence / 100.0, 2),
                'device_name': self.device_name,
                'detected_at': date_time,
                'audio_detected': False,
                'camera_detected': True
            }
            files = {
                'image_file': (output_image_path,open(image_path,'rb'),'image/jpeg')
            }
        return image_path, audio_path, date_time, requests.post(link, headers=headers, files=files, data={'payload': json.dumps(payload)})

class Files:
    def __init__(self):
        pass

    def moveToCompletedDir(self, audio_path, image_path):
        audio_path = audio_path[2::]
        image_path = image_path[2::]
        audio_file_destination_path = audio_path.replace('pending','completed')
        image_file_destination_path = image_path.replace('pending','completed')
        shutil.move(image_path, image_file_destination_path)
        logger.info('Moved to completed: ' + str(image_file_destination_path))
        shutil.move(audio_path, audio_file_destination_path)
        logger.info('Moved to completed: ' + str(audio_file_destination_path))

class GPS:
    def __init__(self):
        try:
            self.serial_port_setup()
            if self.activateGPS():
                self.isGPSActivated = True
                logger.info('GPS activated')
                print('GPS activated')
            else:
                self.isGPSActivated = False
                logger.warning('GPS module not responding')
                print('GPS module not responding')
        except:
            self.isGPSActivated = False
            logger.warning('GPS module not responding')
            print('GPS module not responding')

    def serial_port_setup(self):
        self.ser0 = serial.Serial('/dev/ttyAMA0', 115200, timeout = 1)
        self.ser2 = serial.Serial('/dev/ttyUSB2', 115200, timeout = 1)
        self.ser0.reset_input_buffer()
        self.ser0.reset_output_buffer()
        self.ser2.reset_input_buffer()
        self.ser2.reset_output_buffer()

    def activateGPS(self):
        gps_value = self.getGPSInFo()
        if len(gps_value) == 0:
            self.ser0.write('AT+CGPS=1\r\n'.encode())
            self.ser0.close()
            if len(self.getGPSInFo()) == 0:
                self.ser2.close()
                return False
            else:
                return True
        else:
            return True
    
    def getGPSInFo(self):
        self.ser2.write('AT+CGPSINFO\r\n'.encode())
        line = 0
        while self.ser2.in_waiting > 0:
            line = self.ser2.readline().decode('utf-8').rstrip()
            if 'GPGGA' in line:
                return self.GPSDataAbstraction(line)
        return ''
            
    def GPSDataAbstraction(self, line):
        try:
            element = line.split(',')
            if '' not in element[2:6]:
                latitude_degree = math.trunc(float(element[2]) / 100)
                latitude_decimal = float(element[2]) % 100.00 / 60.0
                latitude = round(latitude_degree + latitude_decimal,5)
                longitude_degree = math.trunc(float(element[4]) / 100)
                longitude_decimal = float(element[4]) % 100.00 / 60.0
                longitude = round(longitude_degree + longitude_decimal,5)
                return str(latitude) + "°" + str(element[3]) + ' ' + str(longitude) + "°" + str(element[5])
            else:
                logger.warning('GPS signal lost')
                print('GPS signal lost')
                return ' '
        except:
            logger.warning('GPS signal lost')
            print('GPS signal lost')
            return ''

    def deactivateGPS(self):
        self.ser2.close()
        self.ser0.open()
        self.ser0.reset_input_buffer()
        self.ser0.reset_output_buffer()
        self.ser0.write('AT+CGPS=0\r\n'.encode())
        self.ser0.close()

class Battery:
    def __init__(self):
        self.i2c = busio.I2C(3,2)
        self.ads = ADS.ADS1115(self.i2c)
        self.channel = AnalogIn(self.ads, ADS.P1)
        self.in_min = 1.769
        self.in_max = 2.477
        self.out_min = 0
        self.out_max = 100

    def get_battery_level(self, voltage_input):
        return str(round((voltage_input - self.in_min) * (self.out_max - self.out_min) / (self.in_max - self.in_min) + self.out_min,0))
###################################################################   Class   ###################################################################


###################################################################   Main Flow   ###################################################################
if __name__ == '__main__':

    ###################################################################   Setup   ###################################################################
    logger.info('System setup initiated')
    print('System setup initiated')

    # pir = MotionSensor(17)
    camera = Image()
    skipping_timer = camera.updateTimer()
    logger.info('Camera initialized | Resolution: ' + str(camera.resolution))
    microphone = Audio()
    logger.info('Microphone initialized | Rate: ' + str(microphone.sampling_rate))
    api = API()
    lora = LoRa()
    logger.info('LoRa initialized')
    files = Files()
    gps = GPS()
    battery = Battery()

    logger.info('All components initialized successfully')
    print('All components initialized successfully')
    ###################################################################   Setup   ###################################################################

    try:
        # Superloop
        while True:
            # Handshaking mechanism (Done it every 1 hour and during first startup)
            if time.time() - api.handshake_timer >= api.handshake_duration or api.handshake_timer == 0:
                try:
                    response = api.handshake()
                except:
                    response.status_code = 400
                
                # Handshake result (If failed then sent LoRa message)
                if response.status_code == 200:
                    logger.info('API handshake successful | Response: 200')
                    print('API handshake successful | Response: 200')
                    print(response.content)
                else:
                    logger.info('API handshake failed | Attempting LoRa fallback | Status: ' + str(response.status_code))
                    print('API handshake failed | Attempting LoRa fallback | Status: ' + str(response.status_code))
                    lora.sendLoRaMessage('/handshake/')
                    logger.info('LoRa packet sent | Type: handshake | Content: /handshake/')
                    print('LoRa packet sent | Type: handshake | Content: /handshake/')

                api.handshake_timer = time.time()

                # Print out GPS data only when GPS is activated
                if gps.isGPSActivated:
                    gps_data = gps.getGPSInFo()
                    logger.info('GPS coordinates updated | Content: ' + gps_data)
                    print('GPS coordinates updated | Content: ' + gps_data)

                logger.info('Battery level: ' + battery.get_battery_level(battery.channel.voltage) + '%')
                print('Battery level: ' + battery.get_battery_level(battery.channel.voltage) + '%')

            # if pir.triggered:
            #     isElephantDetected = True

            image_confidence = 0
            is_sound_flow_done = False

            # Image flow (Include time skipping)
            skipping_timer, date_time, image_confidence = camera.imageFlow(skipping_timer)

            
            # Audio flow
            if image_confidence != 0:
                stream = microphone.startRecording()
                sound_frame = microphone.audioRecording(stream)
                microphone.stopRecording(stream)
                audio_file_path = microphone.saveWavFile(date_time, sound_frame)
                mel_spectrogram = microphone.convertToSpectrogram(audio_file_path)
                audio_confidence = microphone.audioDetection(mel_spectrogram)
                is_sound_flow_done = True

            if is_sound_flow_done:
                try:
                    image_path, audio_path, date_time, response = api.detect(image_confidence,audio_confidence,date_time)
                except:
                    response.status_code = 400

                if response.status_code == 200:
                    logger.info('Detection data uploaded | Response: 200')
                    print('Detection data uploaded | Response: 200')
                    print(response.content)
                    files.moveToCompletedDir(audio_path, image_path)
                else:
                    logger.warning('API upload failed | Attempting LoRa fallback | Status: ' + str(response.status_code))
                    print('API upload failed | Attempting LoRa fallback | Status: ' + str(response.status_code))
                    date_time = time.strptime(date_time, "%Y_%m_%d_%H_%M_%S")
                    date_time = time.strftime("%Y-%m-%dT%H:%M:%S",date_time)
                    lora_message = '/detect/' + str(round(audio_confidence / 100.0, 2)) + '/' + str(round(image_confidence/100.0, 2)) + '/' + date_time + '/'
                    lora.sendLoRaMessage(lora_message)
                    logger.info('LoRa packet sent | Type: detect | Content: ' + lora_message)
                    print('LoRa packet sent | Type: detect | Content: ' + lora_message)

            battery_level = float(battery.get_battery_level(battery.channel.voltage))
            if battery_level <= 5 and battery_level >= 0:
                break
    except KeyboardInterrupt:
        pass

    logger.info('System shutdown initiated')

    camera.camera.stop()
    microphone.microphone.terminate()
    cv2.destroyAllWindows()
    if gps.isGPSActivated:
        gps.deactivateGPS()
    
    logger.info('System shutdown successfully')
    print('System off')
###################################################################   Main Flow   ###################################################################