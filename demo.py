#!/usr/bin/env python3
"""
간단한 데모 스크립트
샘플 데이터로 전체 시스템을 빠르게 테스트합니다.
"""

import os
import sys
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import MIDITokenizer, MIDIPreprocessor
from src.model import MusicGPT, MusicGPTConfig
from src.trainer import MusicGPTTrainer, create_data_loaders, get_default_config
from src.music_generator import MusicGenerator, MusicEvaluator
import json

def create_sample_data():
    """샘플 MIDI 데이터 생성"""
    print("샘플 데이터 생성 중...")
    
    # 디렉토리 생성
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('demo_output', exist_ok=True)
    
    # 토크나이저 생성
    tokenizer = MIDITokenizer()
    
    # 간단한 멜로디 패턴 생성 (C major scale)
    sample_data = []
    
    for i in range(20):  # 20개 샘플 시퀀스
        tokens = [tokenizer.special_tokens['[BOS]']]
        
        # C major scale: C, D, E, F, G, A, B
        scale_notes = [60, 62, 64, 65, 67, 69, 71]  # MIDI note numbers
        
        # 간단한 멜로디 생성
        for j in range(32):  # 32개 음표
            # 음표 선택 (스케일 내에서)
            note_idx = (i + j) % len(scale_notes)
            pitch = scale_notes[note_idx]
            
            # 가끔 옥타브 변경
            if j % 8 == 0 and j > 0:
                pitch += 12 if j % 16 == 0 else -12
                pitch = max(48, min(84, pitch))  # 범위 제한
            
            # 토큰 추가
            tokens.append(tokenizer.pitch_to_token(pitch))
            tokens.append(tokenizer.duration_to_token(0.25))  # 16분음표
            
            # 가끔 쉼표 추가
            if j % 4 == 3:
                tokens.append(tokenizer.time_shift_to_token(0.125))  # 짧은 쉼표
        
        tokens.append(tokenizer.special_tokens['[EOS]'])
        
        sample_data.append({
            'file_path': f'sample_{i:03d}.mid',
            'tokens': tokens,
            'length': len(tokens)
        })
    
    # 데이터 저장
    with open('data/processed/processed_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    # 토크나이저 설정 저장
    tokenizer_config = {
        'max_pitch': tokenizer.max_pitch,
        'max_duration': tokenizer.max_duration,
        'max_time_shift': tokenizer.max_time_shift,
        'vocab_size': tokenizer.vocab_size,
        'special_tokens': tokenizer.special_tokens,
        'token_types': tokenizer.token_types
    }
    
    with open('data/processed/tokenizer_config.json', 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"샘플 데이터 생성 완료: {len(sample_data)}개 시퀀스")
    return tokenizer

def quick_train_demo(tokenizer):
    """빠른 훈련 데모"""
    print("\n빠른 훈련 데모 시작...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 작은 모델 설정 (빠른 테스트용)
    model_config = MusicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,  # 작은 모델
        n_heads=4,
        n_layers=2,
        max_len=256
    )
    
    # 모델 생성
    model = MusicGPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers
    )
    
    print(f"모델 생성 완료: {sum(p.numel() for p in model.parameters()):,} 파라미터")
    
    # 데이터 로더 생성
    train_loader, _ = create_data_loaders(
        train_data_file='data/processed/processed_data.json',
        batch_size=4,  # 작은 배치
        max_length=256,
        num_workers=0  # 단순화
    )
    
    # 훈련 설정
    train_config = get_default_config()
    train_config.update({
        'learning_rate': 1e-3,  # 높은 학습률로 빠른 학습
        'max_epochs': 5,  # 짧은 훈련
        'checkpoint_dir': 'demo_output/checkpoints',
        'log_dir': 'demo_output/logs',
        'save_interval': 2
    })
    
    # 디렉토리 생성
    os.makedirs(train_config['checkpoint_dir'], exist_ok=True)
    os.makedirs(train_config['log_dir'], exist_ok=True)
    
    # 트레이너 생성 및 훈련
    trainer = MusicGPTTrainer(model, train_config, device)
    trainer.train(train_loader)
    
    print("빠른 훈련 완료")
    return model

def generate_demo_music(model, tokenizer):
    """데모 음악 생성"""
    print("\n데모 음악 생성 중...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 생성기 생성
    generator = MusicGenerator(model, tokenizer, device)
    
    # 음악 생성
    generated_files = generator.generate_music(
        output_dir='demo_output/generated_music',
        num_pieces=3,
        piece_length=128,  # 짧은 조각
        temperature=1.0,
        top_k=30,
        top_p=0.8,
        tempo=120
    )
    
    print(f"데모 음악 생성 완료: {len(generated_files)}개 파일")
    
    # 간단한 평가
    if generated_files:
        evaluator = MusicEvaluator(tokenizer)
        preprocessor = MIDIPreprocessor(tokenizer)
        
        print("\n생성된 음악 평가:")
        for file_path in generated_files:
            try:
                tokens = preprocessor.process_midi_file(file_path)
                if tokens:
                    metrics = evaluator.evaluate_comprehensive(tokens)
                    filename = os.path.basename(file_path)
                    print(f"  {filename}: 점수 {metrics['overall_score']:.3f}")
            except Exception as e:
                print(f"  평가 실패 {file_path}: {e}")
    
    return generated_files

def main():
    print("=== AI 음악 생성 시스템 데모 ===")
    print("이 데모는 샘플 데이터로 전체 시스템을 빠르게 테스트합니다.\n")
    
    try:
        # 1. 샘플 데이터 생성
        tokenizer = create_sample_data()
        
        # 2. 빠른 훈련
        model = quick_train_demo(tokenizer)
        
        # 3. 음악 생성
        generated_files = generate_demo_music(model, tokenizer)
        
        print("\n=== 데모 완료 ===")
        print(f"생성된 파일들:")
        for file_path in generated_files:
            print(f"  - {file_path}")
        
        print("\n데모 결과는 demo_output/ 디렉토리에서 확인할 수 있습니다.")
        print("\n전체 시스템을 사용하려면 다음 명령을 실행하세요:")
        print("  python main.py --mode full --epochs 50 --rl_iterations 1000")
        
    except Exception as e:
        print(f"\n데모 실행 중 오류 발생: {e}")
        print("\n문제 해결을 위해 다음을 확인해주세요:")
        print("1. 가상환경이 활성화되어 있는지")
        print("2. 필요한 패키지가 설치되어 있는지 (pip install -r requirements.txt)")
        print("3. 충분한 디스크 공간이 있는지")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()