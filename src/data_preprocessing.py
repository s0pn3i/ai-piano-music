import os
import json
import numpy as np
import pretty_midi
from miditoolkit import MidiFile
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm

class MIDITokenizer:
    """MIDI 데이터를 토큰으로 변환하는 클래스"""
    
    def __init__(self, max_pitch=127, max_duration=32, max_time_shift=100):
        self.max_pitch = max_pitch
        self.max_duration = max_duration
        self.max_time_shift = max_time_shift
        
        # 특수 토큰 정의
        self.special_tokens = {
            '[PAD]': 0,
            '[BOS]': 1,
            '[EOS]': 2,
            '[UNK]': 3
        }
        
        # 토큰 타입별 시작 인덱스
        self.token_types = {
            'special': 0,
            'pitch': 4,
            'duration': 4 + max_pitch + 1,
            'time_shift': 4 + max_pitch + 1 + max_duration + 1
        }
        
        # 전체 어휘 크기
        self.vocab_size = (4 + max_pitch + 1 + max_duration + 1 + max_time_shift + 1)
        
        # 역변환을 위한 매핑
        self.id_to_token = self._create_id_to_token_mapping()
    
    def _create_id_to_token_mapping(self) -> Dict[int, str]:
        """토큰 ID를 토큰 문자열로 변환하는 매핑 생성"""
        mapping = {}
        
        # 특수 토큰
        for token, idx in self.special_tokens.items():
            mapping[idx] = token
        
        # 피치 토큰
        for pitch in range(self.max_pitch + 1):
            mapping[self.token_types['pitch'] + pitch] = f'PITCH_{pitch}'
        
        # 지속시간 토큰
        for duration in range(self.max_duration + 1):
            mapping[self.token_types['duration'] + duration] = f'DUR_{duration}'
        
        # 시간 이동 토큰
        for time_shift in range(self.max_time_shift + 1):
            mapping[self.token_types['time_shift'] + time_shift] = f'TIME_{time_shift}'
        
        return mapping
    
    def pitch_to_token(self, pitch: int) -> int:
        """피치를 토큰 ID로 변환"""
        if 0 <= pitch <= self.max_pitch:
            return self.token_types['pitch'] + pitch
        return self.special_tokens['[UNK]']
    
    def duration_to_token(self, duration: float) -> int:
        """지속시간을 토큰 ID로 변환 (16분음표 단위로 양자화)"""
        # 16분음표를 1로 하는 양자화
        quantized_duration = min(int(duration * 4), self.max_duration)
        return self.token_types['duration'] + quantized_duration
    
    def time_shift_to_token(self, time_shift: float) -> int:
        """시간 이동을 토큰 ID로 변환"""
        quantized_shift = min(int(time_shift * 4), self.max_time_shift)
        return self.token_types['time_shift'] + quantized_shift
    
    def token_to_pitch(self, token_id: int) -> Optional[int]:
        """토큰 ID를 피치로 변환"""
        if self.token_types['pitch'] <= token_id < self.token_types['duration']:
            return token_id - self.token_types['pitch']
        return None
    
    def token_to_duration(self, token_id: int) -> Optional[float]:
        """토큰 ID를 지속시간으로 변환"""
        if self.token_types['duration'] <= token_id < self.token_types['time_shift']:
            return (token_id - self.token_types['duration']) / 4.0
        return None
    
    def token_to_time_shift(self, token_id: int) -> Optional[float]:
        """토큰 ID를 시간 이동으로 변환"""
        if token_id >= self.token_types['time_shift']:
            return (token_id - self.token_types['time_shift']) / 4.0
        return None

class MIDIPreprocessor:
    """MIDI 파일을 전처리하는 클래스"""
    
    def __init__(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer
    
    def load_midi_file(self, file_path: str) -> Optional[pretty_midi.PrettyMIDI]:
        """MIDI 파일 로드"""
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            return midi_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_notes(self, midi_data: pretty_midi.PrettyMIDI) -> List[Dict]:
        """MIDI 데이터에서 음표 정보 추출"""
        notes = []
        
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                notes.append({
                    'pitch': note.pitch,
                    'start': note.start,
                    'end': note.end,
                    'duration': note.end - note.start,
                    'velocity': note.velocity
                })
        
        # 시작 시간으로 정렬
        notes.sort(key=lambda x: x['start'])
        return notes
    
    def notes_to_tokens(self, notes: List[Dict]) -> List[int]:
        """음표 리스트를 토큰 시퀀스로 변환"""
        tokens = [self.tokenizer.special_tokens['[BOS]']]
        
        current_time = 0.0
        
        for note in notes:
            # 시간 이동 토큰 추가
            time_shift = note['start'] - current_time
            if time_shift > 0:
                tokens.append(self.tokenizer.time_shift_to_token(time_shift))
            
            # 피치 토큰 추가
            tokens.append(self.tokenizer.pitch_to_token(note['pitch']))
            
            # 지속시간 토큰 추가
            tokens.append(self.tokenizer.duration_to_token(note['duration']))
            
            current_time = note['start']
        
        tokens.append(self.tokenizer.special_tokens['[EOS]'])
        return tokens
    
    def tokens_to_notes(self, tokens: List[int]) -> List[Dict]:
        """토큰 시퀀스를 음표 리스트로 변환"""
        notes = []
        current_time = 0.0
        current_pitch = None
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # 특수 토큰 처리
            if token in [self.tokenizer.special_tokens['[BOS]'], 
                        self.tokenizer.special_tokens['[EOS]'],
                        self.tokenizer.special_tokens['[PAD]']]:
                i += 1
                continue
            
            # 시간 이동 토큰
            time_shift = self.tokenizer.token_to_time_shift(token)
            if time_shift is not None:
                current_time += time_shift
                i += 1
                continue
            
            # 피치 토큰
            pitch = self.tokenizer.token_to_pitch(token)
            if pitch is not None:
                current_pitch = pitch
                i += 1
                continue
            
            # 지속시간 토큰
            duration = self.tokenizer.token_to_duration(token)
            if duration is not None and current_pitch is not None:
                notes.append({
                    'pitch': current_pitch,
                    'start': current_time,
                    'end': current_time + duration,
                    'duration': duration,
                    'velocity': 80  # 기본 벨로시티
                })
                current_pitch = None
                i += 1
                continue
            
            i += 1
        
        return notes
    
    def process_midi_file(self, file_path: str) -> Optional[List[int]]:
        """MIDI 파일을 처리하여 토큰 시퀀스 반환"""
        midi_data = self.load_midi_file(file_path)
        if midi_data is None:
            return None
        
        notes = self.extract_notes(midi_data)
        if not notes:
            return None
        
        tokens = self.notes_to_tokens(notes)
        return tokens
    
    def process_midi_directory(self, directory_path: str, output_file: str, max_files: Optional[int] = None):
        """디렉토리의 모든 MIDI 파일을 처리하여 데이터셋 생성"""
        midi_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, file))
        
        if max_files:
            midi_files = midi_files[:max_files]
        
        processed_data = []
        
        for file_path in tqdm(midi_files, desc="Processing MIDI files"):
            tokens = self.process_midi_file(file_path)
            if tokens:
                processed_data.append({
                    'file_path': file_path,
                    'tokens': tokens,
                    'length': len(tokens)
                })
        
        # JSON으로 저장
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed {len(processed_data)} files and saved to {output_file}")
        return processed_data

def create_pytorch_dataset(data_file: str, max_length: int = 512) -> torch.utils.data.Dataset:
    """PyTorch 데이터셋 생성"""
    
    class MIDIDataset(torch.utils.data.Dataset):
        def __init__(self, data_file: str, max_length: int):
            with open(data_file, 'r') as f:
                self.data = json.load(f)
            self.max_length = max_length
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            tokens = self.data[idx]['tokens']
            
            # 패딩 또는 자르기
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            else:
                tokens = tokens + [0] * (self.max_length - len(tokens))  # PAD 토큰으로 패딩
            
            return {
                'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
                'labels': torch.tensor(tokens[1:], dtype=torch.long),
                'attention_mask': torch.tensor([1 if t != 0 else 0 for t in tokens[:-1]], dtype=torch.long)
            }
    
    return MIDIDataset(data_file, max_length)

if __name__ == "__main__":
    # 사용 예시
    tokenizer = MIDITokenizer()
    preprocessor = MIDIPreprocessor(tokenizer)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Token types: {tokenizer.token_types}")