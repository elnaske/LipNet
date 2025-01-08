import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import cv2
import dlib
import numpy as np

class LipNetDataset(Dataset):
    def __init__(self, file_paths, vocab):
        self.vids = []
        self.texts = []

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        self.char_to_num = {c: i for i, c in enumerate(vocab)}

        for f in tqdm(file_paths):

            vid = self.load_video(f)
            if vid is not None:
                text = self.get_transcript(f)
                text = torch.tensor([self.char_to_num[c] for c in text])

                self.vids.append(vid)

                self.texts.append(text)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.vids[idx], self.texts[idx]
    
    def extract_mouth(self, frame, landmarks):
        mouth_points = landmarks[48:68]
        x_min = np.min(mouth_points[:, 0])
        x_max = np.max(mouth_points[:, 0])
        y_min = np.min(mouth_points[:, 1])
        y_max = np.max(mouth_points[:, 1])

        x = int((x_min + x_max) / 2)
        y = int((y_min + y_max) / 2)

        return x, y

    def get_mouth_coords(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error opening video file: {path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            print(f"Video is empty: {path}")
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, (frame_count // 2))

        faces = []
        while not len(faces) > 0:
            ret, frame = cap.read()
            if ret is not None:
                faces = self.detector(frame)
        face = faces[0]
        landmarks = self.predictor(frame, face)
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        x, y = self.extract_mouth(frame, landmarks_array)
        cap.release()
        return x, y
    
    def load_video(self, path):
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            print(f"Error opening video file: {path}")
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count == 0:
            print(f"Video is empty: {path}")
            cap.release()
            return None

        frames = []

        mouth_coords = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            if ret is not None:
                faces = self.detector(frame)
                if len(faces) > 0:
                    faces = self.detector(frame)
                    face = faces[0]
                    landmarks = self.predictor(frame, face)
                    landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
                    x, y = self.extract_mouth(frame, landmarks_array)
                    mouth_coords.append((x,y))
                elif mouth_coords:
                    x, y = mouth_coords[-1]
                else:
                    print(f"Couldn't extract mouth for {path}, skipping...")
                    return None

                x_min, x_max = int(x - 30), int(x + 30)
                y_min, y_max = int(y - 20), int(y + 20)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                cropped_frame = frame[y_min:y_max, x_min:x_max]
                frame_tensor = torch.tensor(cropped_frame, dtype=torch.float32)

                frames.append(frame_tensor)

        cap.release()

        frames = torch.stack(frames, dim=0).unsqueeze(0)

        mean = frames.mean()
        std = frames.std()

        # Normalize the frames
        normalized_frames = (frames - mean) / std

        # Pad or truncate to 75 frames
        no_frames = frames.size(1)
        if no_frames < 75:
            pad_len = 75 - no_frames
            normalized_frames = torch.nn.functional.pad(normalized_frames, (0, 0, 0, 0, 0, pad_len))
        elif no_frames > 75:
            normalized_frames = normalized_frames[:, :75, :, :]

        if not normalized_frames.shape == torch.Size([1, 75, 40, 60]):
            print(f"Wrong shape for {path}, skipping...")
            return None

        return normalized_frames

def collate_fn(batch):
    inputs, labels = zip(*batch)

    # Get lengths of inputs and labels
    input_lengths = torch.tensor([x.size(1) for x in inputs], dtype=torch.long)
    label_lengths = torch.tensor([len(y) for y in labels], dtype=torch.long)

    # Concatenate labels (CTCLoss expects a single flattened tensor)
    labels_concatenated = torch.cat(labels)
    inputs_stacked = torch.stack(inputs)

    return inputs_stacked, labels_concatenated, input_lengths, label_lengths
