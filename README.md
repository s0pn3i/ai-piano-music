# AI Piano Music Generation System

A comprehensive AI system for generating piano music using GPT-based models with supervised learning and reinforcement learning.

## Features

- **Data Preprocessing**: MIDI file processing and tokenization
- **GPT Model**: Transformer-based music generation model
- **Supervised Learning**: Train on existing MIDI datasets
- **Reinforcement Learning**: Improve music quality through RL
- **Music Generation**: Generate new piano compositions
- **MIDI Player**: Built-in MIDI file player with multiple backends

## Project Structure

```bash
# 가상환경 생성 및 활성화
python3 -m venv music-env
source music-env/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행

```bash
# 샘플 데이터로 전체 시스템 테스트
python main.py --mode full --epochs 10 --rl_iterations 100
```

### 3. 단계별 실행

```bash
# 1. 데이터 전처리 (MIDI 파일이 있는 경우)
python main.py --mode preprocess --midi_dir /path/to/midi/files

# 2. 지도학습
python main.py --mode train --epochs 50 --batch_size 8

# 3. 강화학습
python main.py --mode rl --rl_iterations 1000

# 4. 음악 생성
python main.py --mode generate --num_pieces 10 --temperature 1.2
```

## 📁 프로젝트 구조

```
music/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # MIDI 데이터 전처리
│   ├── model.py                 # GPT 기반 음악 생성 모델
│   ├── trainer.py               # 지도학습 트레이너
│   ├── rl_trainer.py           # PPO 강화학습 트레이너
│   └── music_generator.py       # 음악 생성 및 평가
├── data/
│   ├── raw/                     # 원본 MIDI 파일
│   └── processed/               # 전처리된 데이터
├── checkpoints/                 # 지도학습 모델 체크포인트
├── rl_checkpoints/             # 강화학습 모델 체크포인트
├── logs/                       # 훈련 로그
├── generated_music/            # 생성된 음악 파일
├── evaluation_results/         # 평가 결과
├── main.py                     # 메인 실행 스크립트
├── requirements.txt            # 의존성 목록
└── README.md                   # 프로젝트 설명
```

## 🔧 주요 구성 요소

### 1. 데이터 전처리 (`data_preprocessing.py`)

- **MIDITokenizer**: MIDI 데이터를 토큰으로 변환
  - Pitch 토큰: 음높이 (0-127)
  - Duration 토큰: 지속시간 (16분음표 단위)
  - Time Shift 토큰: 시간 이동
  - 특수 토큰: [BOS], [EOS], [PAD], [UNK]

- **MIDIPreprocessor**: MIDI 파일 처리 및 토큰 시퀀스 생성

### 2. 모델 구조 (`model.py`)

- **MusicGPT**: GPT 기반 음악 생성 모델
  - Multi-head Self-attention
  - Position Encoding
  - Causal Masking (미래 토큰 차단)
  - Residual Connection & Layer Normalization

### 3. 지도학습 (`trainer.py`)

- **MusicGPTTrainer**: 지도학습 훈련 관리
  - AdamW Optimizer
  - Cosine Annealing 학습률 스케줄러
  - Gradient Clipping
  - TensorBoard 로깅

### 4. 강화학습 (`rl_trainer.py`)

- **PPOTrainer**: PPO 기반 강화학습
- **MusicRewardCalculator**: 음악 품질 보상 계산
  - 화성 일관성 (Harmony Consistency)
  - 음 다양성 (Pitch Diversity)
  - 리듬 구조 (Rhythm Structure)
  - 중복 회피 (Repetition Penalty)

### 5. 음악 생성 (`music_generator.py`)

- **MusicGenerator**: 훈련된 모델로 음악 생성
  - Temperature Sampling
  - Top-k, Top-p Sampling
  - MIDI 파일 출력

- **MusicEvaluator**: 생성된 음악 품질 평가

## ⚙️ 설정 옵션

### 모델 하이퍼파라미터

```bash
python main.py --mode train \
  --d_model 512 \
  --n_heads 8 \
  --n_layers 6 \
  --max_length 512 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --epochs 100
```

### 강화학습 설정

```bash
python main.py --mode rl \
  --rl_learning_rate 1e-5 \
  --rl_iterations 1000 \
  --rollout_steps 2048
```

### 음악 생성 설정

```bash
python main.py --mode generate \
  --num_pieces 10 \
  --piece_length 512 \
  --temperature 1.2 \
  --top_k 50 \
  --top_p 0.9 \
  --tempo 120
```

## 📊 평가 지표

생성된 음악은 다음 기준으로 자동 평가됩니다:

1. **Pitch Diversity**: 사용된 음의 다양성
2. **Rhythm Diversity**: 리듬 패턴의 다양성
3. **Scale Consistency**: 음계 일관성 (C major 기준)
4. **Chord Progression**: 화성 진행의 자연스러움
5. **Repetition Rate**: 과도한 반복 패턴 회피
6. **Overall Score**: 종합 점수 (0-1)

## 🎹 MIDI 데이터 준비

### 권장 데이터셋

- **Classical Piano**: 클래식 피아노 곡 MIDI 파일
- **Jazz Standards**: 재즈 스탠다드 곡
- **Pop Piano**: 팝 피아노 편곡

### 데이터 형식

- 파일 형식: `.mid` 또는 `.midi`
- 악기: 피아노 (non-drum instruments)
- 권장 길이: 30초 ~ 5분

### 데이터 배치

```
data/raw/
├── classical/
│   ├── bach_invention1.mid
│   ├── chopin_nocturne.mid
│   └── ...
├── jazz/
│   ├── autumn_leaves.mid
│   ├── blue_moon.mid
│   └── ...
└── pop/
    ├── imagine.mid
    ├── let_it_be.mid
    └── ...
```

## 🔍 모니터링 및 로깅

### TensorBoard 실행

```bash
# 지도학습 로그 확인
tensorboard --logdir logs

# 강화학습 로그 확인
tensorboard --logdir rl_logs
```

### 주요 메트릭

- **Training Loss**: 훈련 손실
- **Perplexity**: 모델 복잡도
- **Learning Rate**: 학습률 변화
- **Reward**: 강화학습 보상
- **Policy Loss**: 정책 손실
- **Value Loss**: 가치 함수 손실

## 🚨 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python main.py --batch_size 4
   ```

2. **훈련 속도 느림**
   ```bash
   # 워커 수 조정
   python main.py --num_workers 2
   ```

3. **생성 품질 낮음**
   ```bash
   # 온도 조정
   python main.py --mode generate --temperature 0.8
   ```

### 디버깅 팁

- 로그 파일 확인: `logs/` 디렉토리
- 체크포인트 상태: `checkpoints/` 디렉토리
- 생성된 음악 품질: `evaluation_results/evaluation_results.json`

## 📈 성능 최적화

### GPU 사용

- CUDA 11.8+ 권장
- 최소 8GB VRAM (RTX 3070 이상)
- Mixed Precision 훈련 지원

### 메모리 최적화

```python
# 그래디언트 체크포인팅 활성화
model.gradient_checkpointing_enable()

# 배치 크기 동적 조정
torch.backends.cudnn.benchmark = True
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.

## 🙏 감사의 말

- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: 트랜스포머 모델 구현 참조
- **Pretty MIDI**: MIDI 파일 처리
- **Stable Baselines3**: 강화학습 알고리즘

## 📞 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 이용해 주세요.

---

**Happy Music Generation! 🎵**