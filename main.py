#!/usr/bin/env python3
"""
피아노 음악 MIDI 데이터 기반 AI 음악 생성 시스템

전체 파이프라인:
1. 가상환경 설정 (수동)
2. MIDI 데이터 전처리
3. GPT 모델 구성 및 지도학습
4. PPO 기반 강화학습
5. 음악 생성 및 평가
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import MIDITokenizer, MIDIPreprocessor, create_pytorch_dataset
from src.model import MusicGPT, MusicGPTConfig, create_model
from src.trainer import MusicGPTTrainer, create_data_loaders, get_default_config
from src.rl_trainer import PPOTrainer, get_rl_config
from src.music_generator import MusicGenerator, MusicEvaluator, load_trained_model

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        'data/raw',
        'data/processed', 
        'checkpoints',
        'rl_checkpoints',
        'logs',
        'rl_logs',
        'generated_music',
        'evaluation_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def preprocess_data(args):
    """MIDI 데이터 전처리"""
    print("=== MIDI 데이터 전처리 시작 ===")
    
    # 토크나이저 생성
    tokenizer = MIDITokenizer(
        max_pitch=args.max_pitch,
        max_duration=args.max_duration,
        max_time_shift=args.max_time_shift
    )
    
    # 전처리기 생성
    preprocessor = MIDIPreprocessor(tokenizer)
    
    print(f"토크나이저 어휘 크기: {tokenizer.vocab_size}")
    
    # MIDI 파일 처리
    if os.path.exists(args.midi_dir):
        processed_data = preprocessor.process_midi_directory(
            directory_path=args.midi_dir,
            output_file='data/processed/processed_data.json',
            max_files=args.max_files
        )
        
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
        
        print(f"전처리 완료: {len(processed_data)}개 파일 처리됨")
        return tokenizer
    else:
        print(f"MIDI 디렉토리를 찾을 수 없습니다: {args.midi_dir}")
        print("샘플 데이터로 진행합니다...")
        
        # 샘플 데이터 생성
        sample_data = []
        for i in range(10):
            # 간단한 샘플 토큰 시퀀스 생성
            tokens = [tokenizer.special_tokens['[BOS]']]
            for j in range(50):
                tokens.extend([
                    tokenizer.pitch_to_token(60 + (j % 12)),  # C4부터 시작하는 음계
                    tokenizer.duration_to_token(0.25),  # 16분음표
                    tokenizer.time_shift_to_token(0.25)  # 16분음표 간격
                ])
            tokens.append(tokenizer.special_tokens['[EOS]'])
            
            sample_data.append({
                'file_path': f'sample_{i}.mid',
                'tokens': tokens,
                'length': len(tokens)
            })
        
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

def train_supervised(args, tokenizer):
    """지도학습 훈련"""
    print("=== 지도학습 훈련 시작 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    # 모델 설정
    model_config = MusicGPTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_length
    )
    
    # 모델 생성
    model = create_model(model_config)
    print(f"모델 생성 완료: {sum(p.numel() for p in model.parameters()):,} 파라미터")
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(
        train_data_file='data/processed/processed_data.json',
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    print(f"훈련 데이터: {len(train_loader)} 배치")
    
    # 훈련 설정
    train_config = get_default_config()
    train_config.update({
        'learning_rate': args.learning_rate,
        'max_epochs': args.epochs,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    })
    
    # 트레이너 생성 및 훈련
    trainer = MusicGPTTrainer(model, train_config, device)
    
    # 체크포인트에서 재개할 경우
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        trainer.load_checkpoint(args.resume_checkpoint)
        print(f"체크포인트에서 재개: {args.resume_checkpoint}")
    
    trainer.train(train_loader, val_loader)
    
    print("지도학습 훈련 완료")
    return model

def train_reinforcement(args, tokenizer, pretrained_model_path=None):
    """강화학습 훈련"""
    print("=== 강화학습 훈련 시작 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 사전훈련된 모델 로드
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model = load_trained_model(pretrained_model_path, tokenizer, device)
        print(f"사전훈련된 모델 로드: {pretrained_model_path}")
    else:
        # 기본 모델 생성
        model_config = MusicGPTConfig(vocab_size=tokenizer.vocab_size)
        model = create_model(model_config)
        print("새 모델로 강화학습 시작")
    
    # RL 설정
    rl_config = get_rl_config()
    rl_config.update({
        'learning_rate': args.rl_learning_rate,
        'checkpoint_dir': 'rl_checkpoints',
        'log_dir': 'rl_logs'
    })
    
    # RL 트레이너 생성 및 훈련
    rl_trainer = PPOTrainer(model, tokenizer, rl_config, device)
    rl_trainer.train(
        num_iterations=args.rl_iterations,
        rollout_steps=args.rollout_steps
    )
    
    print("강화학습 훈련 완료")
    return model

def generate_music(args, tokenizer, model_path=None):
    """음악 생성"""
    print("=== 음악 생성 시작 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    if model_path and os.path.exists(model_path):
        model = load_trained_model(model_path, tokenizer, device)
        print(f"모델 로드: {model_path}")
    else:
        print("훈련된 모델을 찾을 수 없습니다. 기본 모델을 사용합니다.")
        model_config = MusicGPTConfig(vocab_size=tokenizer.vocab_size)
        model = create_model(model_config)
    
    # 생성기 생성
    generator = MusicGenerator(model, tokenizer, device)
    
    # 음악 생성
    generated_files = generator.generate_music(
        output_dir='generated_music',
        num_pieces=args.num_pieces,
        piece_length=args.piece_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tempo=args.tempo
    )
    
    print(f"음악 생성 완료: {len(generated_files)}개 파일")
    
    # 생성된 음악 평가
    if generated_files:
        evaluate_generated_music(generated_files, tokenizer)
    
    return generated_files

def evaluate_generated_music(generated_files, tokenizer):
    """생성된 음악 평가"""
    print("=== 생성된 음악 평가 시작 ===")
    
    evaluator = MusicEvaluator(tokenizer)
    preprocessor = MIDIPreprocessor(tokenizer)
    
    evaluation_results = []
    
    for file_path in generated_files:
        try:
            # MIDI 파일을 토큰으로 변환
            tokens = preprocessor.process_midi_file(file_path)
            
            if tokens:
                # 평가 수행
                metrics = evaluator.evaluate_comprehensive(tokens)
                metrics['file_path'] = file_path
                evaluation_results.append(metrics)
                
                print(f"{os.path.basename(file_path)}: 전체 점수 {metrics['overall_score']:.3f}")
            
        except Exception as e:
            print(f"평가 실패 {file_path}: {e}")
    
    # 평가 결과 저장
    if evaluation_results:
        with open('evaluation_results/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # 평균 점수 계산
        avg_score = sum(r['overall_score'] for r in evaluation_results) / len(evaluation_results)
        print(f"평균 점수: {avg_score:.3f}")
    
    print("평가 완료")

def load_tokenizer():
    """저장된 토크나이저 설정 로드"""
    config_path = 'data/processed/tokenizer_config.json'
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        tokenizer = MIDITokenizer(
            max_pitch=config['max_pitch'],
            max_duration=config['max_duration'],
            max_time_shift=config['max_time_shift']
        )
        
        print(f"토크나이저 로드 완료: 어휘 크기 {tokenizer.vocab_size}")
        return tokenizer
    else:
        print("토크나이저 설정을 찾을 수 없습니다. 기본 설정을 사용합니다.")
        return MIDITokenizer()

def main():
    parser = argparse.ArgumentParser(description='AI 음악 생성 시스템')
    
    # 모드 선택
    parser.add_argument('--mode', choices=['preprocess', 'train', 'rl', 'generate', 'full'], 
                       default='full', help='실행 모드')
    
    # 데이터 전처리 관련
    parser.add_argument('--midi_dir', type=str, default='data/raw', 
                       help='MIDI 파일 디렉토리')
    parser.add_argument('--max_files', type=int, default=None, 
                       help='처리할 최대 파일 수')
    parser.add_argument('--max_pitch', type=int, default=127, 
                       help='최대 피치 값')
    parser.add_argument('--max_duration', type=int, default=32, 
                       help='최대 지속시간 (16분음표 단위)')
    parser.add_argument('--max_time_shift', type=int, default=100, 
                       help='최대 시간 이동')
    
    # 모델 관련
    parser.add_argument('--d_model', type=int, default=512, 
                       help='모델 차원')
    parser.add_argument('--n_heads', type=int, default=8, 
                       help='어텐션 헤드 수')
    parser.add_argument('--n_layers', type=int, default=6, 
                       help='레이어 수')
    parser.add_argument('--max_length', type=int, default=512, 
                       help='최대 시퀀스 길이')
    
    # 훈련 관련
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='학습률')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='훈련 에포크 수')
    parser.add_argument('--num_workers', type=int, default=4, 
                       help='데이터 로더 워커 수')
    parser.add_argument('--resume_checkpoint', type=str, default=None, 
                       help='재개할 체크포인트 경로')
    
    # 강화학습 관련
    parser.add_argument('--rl_learning_rate', type=float, default=1e-5, 
                       help='강화학습 학습률')
    parser.add_argument('--rl_iterations', type=int, default=1000, 
                       help='강화학습 반복 횟수')
    parser.add_argument('--rollout_steps', type=int, default=2048, 
                       help='롤아웃 스텝 수')
    
    # 생성 관련
    parser.add_argument('--num_pieces', type=int, default=5, 
                       help='생성할 음악 조각 수')
    parser.add_argument('--piece_length', type=int, default=512, 
                       help='음악 조각 길이')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='생성 온도')
    parser.add_argument('--top_k', type=int, default=50, 
                       help='Top-k 샘플링')
    parser.add_argument('--top_p', type=float, default=0.9, 
                       help='Top-p 샘플링')
    parser.add_argument('--tempo', type=int, default=120, 
                       help='MIDI 템포')
    
    args = parser.parse_args()
    
    # 디렉토리 설정
    setup_directories()
    
    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능: {torch.cuda.get_device_name()}")
    else:
        print("CPU 모드로 실행")
    
    # 모드별 실행
    if args.mode == 'preprocess':
        tokenizer = preprocess_data(args)
        
    elif args.mode == 'train':
        tokenizer = load_tokenizer()
        model = train_supervised(args, tokenizer)
        
    elif args.mode == 'rl':
        tokenizer = load_tokenizer()
        # 최고 성능 모델 찾기
        best_model_path = 'checkpoints/best_model.pt'
        model = train_reinforcement(args, tokenizer, best_model_path)
        
    elif args.mode == 'generate':
        tokenizer = load_tokenizer()
        # RL 모델이 있으면 사용, 없으면 지도학습 모델 사용
        rl_model_path = 'rl_checkpoints/rl_checkpoint_900.pt'  # 마지막 RL 체크포인트
        if not os.path.exists(rl_model_path):
            rl_model_path = 'checkpoints/best_model.pt'
        
        generated_files = generate_music(args, tokenizer, rl_model_path)
        
    elif args.mode == 'full':
        # 전체 파이프라인 실행
        print("=== 전체 파이프라인 실행 ===")
        
        # 1. 데이터 전처리
        tokenizer = preprocess_data(args)
        
        # 2. 지도학습
        model = train_supervised(args, tokenizer)
        
        # 3. 강화학습
        best_model_path = 'checkpoints/best_model.pt'
        model = train_reinforcement(args, tokenizer, best_model_path)
        
        # 4. 음악 생성
        rl_model_path = 'rl_checkpoints/rl_checkpoint_900.pt'
        if not os.path.exists(rl_model_path):
            rl_model_path = best_model_path
        
        generated_files = generate_music(args, tokenizer, rl_model_path)
        
        print("=== 전체 파이프라인 완료 ===")
        print(f"생성된 음악 파일: {len(generated_files)}개")
        print("generated_music/ 디렉토리에서 결과를 확인하세요.")

if __name__ == "__main__":
    main()