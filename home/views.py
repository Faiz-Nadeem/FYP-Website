from django.shortcuts import render


from django.core.files.storage import FileSystemStorage
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
import os
import cv2

def index(request):
    return render(request, 'home.html')

def About(request):
    return render(request, 'About.html')

def Articles(request):
    return render(request, 'Articles.html')


def preprocess_data(audio_path):
    segments = []
    segments_2 = []

    # print('segment length 1: ', len(segments))
    try:
        num_samples = 88200

        samples, sample_rate = sf.read(audio_path, dtype='float32')
        # print(sample_rate)
        # print(len(samples))

        if len(samples.shape) == 2 and samples.shape[1] == 2:
            # Convert stereo to mono by averaging channels
            print('Sound is stereo')
            samples = np.mean(samples, axis=1)

        if sample_rate != 44100:
            print('Different sample rate')
            samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=44100)
            sample_rate = 44100

        total_samples = len(samples)
        # print('total samples: ', total_samples)
        # print(sample_rate)
        # print(samples.shape)
        loop_num = total_samples - (total_samples % num_samples)
        # print('fadsf: ', loop_num)
        loop_num = loop_num / num_samples
        loop_num = int(loop_num * 2)
        # print('loop number: ', loop_num)

        for i in range(loop_num + 1):
            # print(i)
            start = i * num_samples
            end = start + num_samples
            segment = samples[start:end]
            # print(segment.shape)
            segments.append(segment)

        # for i in range(len(segments)):
        #     print(segments[i])
        #     print('---')

        # print('segment_2 length: ', len(segments_2))

        # print(type(segments))
        for i in range(len(segments)):
            if len(segments[i]) > 0 and len(segments[i]) == 88200:
                # print(' is not null')
                # print(segments[i].shape)

                segments_2.append(segments[i])
            else:
                print('is Null')

        # for i in range(len(segments_2)):
        #     if len(segments_2[i]) < 88200:
        #         print('short segment')
        #     print(len(segments_2[i]), ' LEngth')
        #     print(segments_2[i].shape)
            # print(segments_2[i])

        # print('segment_2 length: ', len(segments_2))

        # print('executing trimming code!')
        segments_2 = [librosa.effects.trim(segment, top_db=25)[0] for segment in segments_2]
        # print('trim code executed')
        #
        # for i in range(len(segments_2)):
        #     if len(segments_2[i]) < 88200:
        #         print('short segment')
        #     print(len(segments_2[i]), ' LEngth')
        #     print(segments_2[i].shape)
        #     print(segments_2[i])

        for i in range(len(segments_2)):
            # print('second i: ', i)
            # print(len(segments_2[i]), ' : ', num_samples)
            if len(segments_2[i]) < num_samples:
                # print(' if block: ', i)
                padding = num_samples - len(segments_2[i])
                segments_2[i] = np.pad(segments_2[i], (0, padding), 'constant')
            else:
                # print('else block: ', i)
                segments_2[i] = segments_2[i][:num_samples]
            # print(segments_2[i].shape)



    except Exception as e:
        print(f"Processing error {audio_path}: {e}")

    # print('segment: ', segments_2[0].shape)
    # print('segment length: ', len(segments_2))
    return segments_2



def extract_features(segments):
    zcr_list = []
    rms_list = []
    mfccs_list = []

    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    for audio in segments:
        if len(audio) == 0:
            continue

        # # Check if segment is long enough
        # if len(audio) < FRAME_LENGTH:
        #     print("Warning: Segment length is shorter than FRAME_LENGTH")
        #     continue

        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)

        zcr_list.append(zcr)
        rms_list.append(rms)
        mfccs_list.append(mfccs)

    return zcr_list, rms_list, mfccs_list



def combine_features(zcr_list, rms_list, mfccs_list):
    zcr_features = np.swapaxes(zcr_list, 1, 2)
    rms_features = np.swapaxes(rms_list, 1, 2)
    mfccs_features = np.swapaxes(mfccs_list, 1, 2)

    X_features = np.concatenate((zcr_features, rms_features, mfccs_features), axis=2)

    return X_features



def detect_ai(request):
    uploaded_file_url = None
    result = None
    # print(f"TensorFlow version: {tf.__version__}")
    # print(f"Keras version: {keras.__version__}")

    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        file_name = uploaded_file.name
        public_url = upload_file_to_firebase(uploaded_file, file_name)
        print(f'File uploaded successfully: {public_url}')
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension = file_extension.lower()

        valid_extensions = {'.mp3', '.png', '.jpeg', '.jpg'}

        if file_extension not in valid_extensions:
            result = "Unsupported file type. Please upload an MP3, PNG, or JPEG file."
            return render(request, 'detect.html', {'result': result})

        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_file_path = fs.path(file_path)
        uploaded_file_url = fs.url(file_path)
        # print(full_file_path)
        if file_extension == '.mp3':

            segments = preprocess_data(full_file_path)
            # print(len(segments))

            zcr_features, rms_features, mfccs_features = extract_features(segments)

            X_features = combine_features(zcr_features, rms_features, mfccs_features)


            model = load_model('AudioDetectionModel.h5')

            prediction = model.predict(X_features)

            print('prediction: ', prediction)
            # print('Type of prediction:', type(prediction))
            # print('Shape of prediction:', prediction.shape)

            first_count = np.sum(prediction[:, 0] > 0.5)
            # Count occurrences where the second element is > 0.5
            second_count = np.sum(prediction[:, 1] > 0.5)

            if first_count > second_count:
                result = full_file_path, "The audio is more likely AI-generated."
            else:
                result = full_file_path, "The audio is more likely not AI-generated."

            # print(result)
        else:
            print('detecting image')
            test_img = cv2.imread(full_file_path)
            test_img = cv2.resize(test_img, (256, 256))
            test_input = test_img.reshape((1, 256, 256, 3))
            image_model = load_model('deepfake_model.h5')
            result_22 = image_model.predict(test_input)
            print(result_22)
            if result_22 == 1:
                result = "The Face image is more likely AI-generated."
            else:
                result = "The Face image is more likely not AI-generated."

    return render(request, 'detect.html', {'result': result, 'uploaded_image': uploaded_file_url})

from firebase_admin import storage
import os

def upload_file_to_firebase(file, file_name):
    bucket = storage.bucket()
    blob = bucket.blob(file_name)
    blob.upload_from_file(file, content_type=file.content_type)
    return blob.public_url