import torch
import numpy as np
import pretty_midi
from typing import List, Dict, Optional, Tuple
import os
from .model import MusicGPT
from .data_preprocessing import MIDITokenizer, MIDIPreprocessor

class MusicGenerator:
    """훈련된 모델을 사용하여 음악을 생성하는 클래스"""
    
    def __init__(self, model: MusicGPT, tokenizer: MIDITokenizer, device: torch.device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.preprocessor = MIDIPreprocessor(tokenizer)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
    
    def generate_sequence(self, prompt: Optional[List[int]] = None, 
                         max_length: int = 512, temperature: float = 1.0,
                         top_k: Optional[int] = 50, top_p: Optional[float] = 0.9,
                         num_sequences: int = 1) -> List[List[int]]:
        """토큰 시퀀스 생성"""
        
        # 프롬프트가 없으면 BOS 토큰으로 시작
        if prompt is None:
            prompt = [self.tokenizer.special_tokens['[BOS]']]
        
        # 배치 차원 추가
        input_ids = torch.tensor([prompt] * num_sequences, dtype=torch.long).to(self.device)
        
        # 생성
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.special_tokens['[PAD]'],
                eos_token_id=self.tokenizer.special_tokens['[EOS]']
            )
        
        # 결과를 리스트로 변환
        sequences = []
        for seq in generated:
            # EOS 토큰까지만 포함
            seq_list = seq.cpu().tolist()
            if self.tokenizer.special_tokens['[EOS]'] in seq_list:
                eos_idx = seq_list.index(self.tokenizer.special_tokens['[EOS]'])
                seq_list = seq_list[:eos_idx + 1]
            sequences.append(seq_list)
        
        return sequences
    
    def tokens_to_midi(self, tokens: List[int], output_path: str, 
                      tempo: int = 120, program: int = 0) -> pretty_midi.PrettyMIDI:
        """토큰 시퀀스를 MIDI 파일로 변환"""
        
        # 토큰을 음표로 변환
        notes = self.preprocessor.tokens_to_notes(tokens)
        
        if not notes:
            print("Warning: No valid notes found in token sequence")
            return None
        
        # MIDI 객체 생성
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        
        # 악기 생성
        instrument = pretty_midi.Instrument(program=program, is_drum=False, name='Piano')
        
        # 음표 추가
        for note_info in notes:
            note = pretty_midi.Note(
                velocity=note_info.get('velocity', 80),
                pitch=int(note_info['pitch']),
                start=float(note_info['start']),
                end=float(note_info['end'])
            )
            instrument.notes.append(note)
        
        midi.instruments.append(instrument)
        
        # MIDI 파일 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        midi.write(output_path)
        
        print(f"MIDI file saved to: {output_path}")
        print(f"Generated {len(notes)} notes, duration: {max(note['end'] for note in notes):.2f} seconds")
        
        return midi
    
    def generate_music(self, output_dir: str, num_pieces: int = 5, 
                      piece_length: int = 512, temperature: float = 1.0,
                      top_k: Optional[int] = 50, top_p: Optional[float] = 0.9,
                      tempo: int = 120) -> List[str]:
        """여러 음악 조각 생성"""
        
        os.makedirs(output_dir, exist_ok=True)
        generated_files = []
        
        print(f"Generating {num_pieces} music pieces...")
        
        for i in range(num_pieces):
            print(f"Generating piece {i+1}/{num_pieces}...")
            
            # 시퀀스 생성
            sequences = self.generate_sequence(
                max_length=piece_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_sequences=1
            )
            
            if sequences and sequences[0]:
                # MIDI 파일로 저장
                output_path = os.path.join(output_dir, f"generated_music_{i+1:03d}.mid")
                
                try:
                    midi = self.tokens_to_midi(sequences[0], output_path, tempo=tempo)
                    if midi is not None:
                        generated_files.append(output_path)
                        print(f"Successfully generated: {output_path}")
                    else:
                        print(f"Failed to generate valid MIDI for piece {i+1}")
                except Exception as e:
                    print(f"Error generating piece {i+1}: {e}")
            else:
                print(f"Failed to generate sequence for piece {i+1}")
        
        print(f"Generated {len(generated_files)} music pieces in {output_dir}")
        return generated_files
    
    def generate_with_prompt(self, prompt_file: str, output_path: str,
                           continuation_length: int = 256, temperature: float = 1.0) -> str:
        """기존 MIDI 파일을 프롬프트로 사용하여 음악 생성"""
        
        # 프롬프트 MIDI 파일 로드 및 토큰화
        prompt_tokens = self.preprocessor.process_midi_file(prompt_file)
        
        if not prompt_tokens:
            raise ValueError(f"Failed to process prompt file: {prompt_file}")
        
        print(f"Loaded prompt with {len(prompt_tokens)} tokens")
        
        # 프롬프트를 사용하여 생성
        sequences = self.generate_sequence(
            prompt=prompt_tokens,
            max_length=len(prompt_tokens) + continuation_length,
            temperature=temperature
        )
        
        if sequences and sequences[0]:
            # MIDI 파일로 저장
            midi = self.tokens_to_midi(sequences[0], output_path)
            if midi is not None:
                print(f"Generated continuation saved to: {output_path}")
                return output_path
        
        raise RuntimeError("Failed to generate music with prompt")

class MusicEvaluator:
    """생성된 음악의 품질을 평가하는 클래스"""
    
    def __init__(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer
    
    def evaluate_diversity(self, tokens: List[int]) -> Dict[str, float]:
        """음악의 다양성 평가"""
        pitches = []
        durations = []
        
        for token in tokens:
            pitch = self.tokenizer.token_to_pitch(token)
            duration = self.tokenizer.token_to_duration(token)
            
            if pitch is not None:
                pitches.append(pitch)
            if duration is not None:
                durations.append(duration)
        
        metrics = {}
        
        # 피치 다양성
        if pitches:
            unique_pitches = len(set(pitches))
            pitch_range = max(pitches) - min(pitches) if len(pitches) > 1 else 0
            metrics['pitch_diversity'] = unique_pitches / 128.0  # MIDI 피치 범위로 정규화
            metrics['pitch_range'] = pitch_range / 128.0
        
        # 리듬 다양성
        if durations:
            unique_durations = len(set(durations))
            metrics['rhythm_diversity'] = min(unique_durations / 10.0, 1.0)  # 최대 10개 리듬으로 정규화
        
        return metrics
    
    def evaluate_repetition(self, tokens: List[int], window_size: int = 8) -> Dict[str, float]:
        """반복 패턴 평가"""
        if len(tokens) < window_size * 2:
            return {'repetition_rate': 0.0}
        
        patterns = {}
        total_windows = len(tokens) - window_size + 1
        
        # 윈도우별 패턴 추출
        for i in range(total_windows):
            pattern = tuple(tokens[i:i + window_size])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # 반복률 계산
        repeated_patterns = sum(1 for count in patterns.values() if count > 1)
        repetition_rate = repeated_patterns / len(patterns) if patterns else 0.0
        
        return {'repetition_rate': repetition_rate}
    
    def evaluate_musical_structure(self, tokens: List[int]) -> Dict[str, float]:
        """음악적 구조 평가"""
        pitches = []
        for token in tokens:
            pitch = self.tokenizer.token_to_pitch(token)
            if pitch is not None:
                pitches.append(pitch % 12)  # 옥타브 정규화
        
        if len(pitches) < 3:
            return {'scale_consistency': 0.0, 'chord_progression': 0.0}
        
        # 음계 일관성 (C major scale 기준)
        c_major = {0, 2, 4, 5, 7, 9, 11}
        in_scale = sum(1 for pitch in pitches if pitch in c_major)
        scale_consistency = in_scale / len(pitches)
        
        # 간단한 코드 진행 평가 (3음씩 그룹화)
        chord_score = 0.0
        chord_count = 0
        
        for i in range(0, len(pitches) - 2, 3):
            chord = set(pitches[i:i+3])
            # 3화음 여부 확인 (간단한 버전)
            if len(chord) >= 2:  # 최소 2개 다른 음
                chord_score += 1
            chord_count += 1
        
        chord_progression = chord_score / chord_count if chord_count > 0 else 0.0
        
        return {
            'scale_consistency': scale_consistency,
            'chord_progression': chord_progression
        }
    
    def evaluate_comprehensive(self, tokens: List[int]) -> Dict[str, float]:
        """종합적인 음악 품질 평가"""
        metrics = {}
        
        # 다양성 평가
        diversity_metrics = self.evaluate_diversity(tokens)
        metrics.update(diversity_metrics)
        
        # 반복 평가
        repetition_metrics = self.evaluate_repetition(tokens)
        metrics.update(repetition_metrics)
        
        # 음악적 구조 평가
        structure_metrics = self.evaluate_musical_structure(tokens)
        metrics.update(structure_metrics)
        
        # 전체 점수 계산
        weights = {
            'pitch_diversity': 0.2,
            'rhythm_diversity': 0.15,
            'scale_consistency': 0.25,
            'chord_progression': 0.2,
            'repetition_penalty': 0.2  # 반복률이 낮을수록 좋음
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            if metric == 'repetition_penalty':
                # 반복률이 낮을수록 좋은 점수
                score = 1.0 - metrics.get('repetition_rate', 0.0)
            else:
                score = metrics.get(metric, 0.0)
            
            overall_score += weight * score
        
        metrics['overall_score'] = overall_score
        
        return metrics

def load_trained_model(checkpoint_path: str, tokenizer: MIDITokenizer, 
                      device: torch.device) -> MusicGPT:
    """훈련된 모델 로드"""
    from .model import MusicGPT, MusicGPTConfig
    
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 설정 추출
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = MusicGPTConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=config.get('d_model', 512),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6)
        )
    else:
        # 기본 설정 사용
        model_config = MusicGPTConfig(vocab_size=tokenizer.vocab_size)
    
    # 모델 생성 및 가중치 로드
    model = MusicGPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded model from {checkpoint_path}")
    return model

if __name__ == "__main__":
    # 사용 예시
    from .data_preprocessing import MIDITokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 토크나이저 생성
    tokenizer = MIDITokenizer()
    
    # 더미 모델로 테스트 (실제로는 훈련된 모델을 로드해야 함)
    from .model import MusicGPT, MusicGPTConfig
    
    config = MusicGPTConfig(vocab_size=tokenizer.vocab_size)
    model = MusicGPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers
    )
    
    # 생성기 생성
    generator = MusicGenerator(model, tokenizer, device)
    
    print("Music generator created successfully")
    
    # 평가기 테스트
    evaluator = MusicEvaluator(tokenizer)
    dummy_tokens = [1, 10, 20, 30, 15, 25, 35, 2]  # 더미 토큰
    metrics = evaluator.evaluate_comprehensive(dummy_tokens)
    print(f"Evaluation metrics: {metrics}")