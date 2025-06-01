import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from .model import MusicGPT
from .data_preprocessing import MIDITokenizer

class MusicRewardCalculator:
    """음악 생성에 대한 보상 계산 클래스"""
    
    def __init__(self, tokenizer: MIDITokenizer):
        self.tokenizer = tokenizer
    
    def calculate_harmony_consistency(self, tokens: List[int]) -> float:
        """화성 일관성 보상 계산"""
        pitches = []
        for token in tokens:
            pitch = self.tokenizer.token_to_pitch(token)
            if pitch is not None:
                pitches.append(pitch % 12)  # 옥타브 정규화
        
        if len(pitches) < 3:
            return 0.0
        
        # 간단한 화성 일관성: 연속된 음들이 조성에 맞는지 확인
        # C major scale: [0, 2, 4, 5, 7, 9, 11]
        c_major = {0, 2, 4, 5, 7, 9, 11}
        
        in_scale_count = sum(1 for pitch in pitches if pitch in c_major)
        consistency_ratio = in_scale_count / len(pitches)
        
        return consistency_ratio
    
    def calculate_pitch_diversity(self, tokens: List[int]) -> float:
        """음 고유도 보상 계산"""
        pitches = set()
        for token in tokens:
            pitch = self.tokenizer.token_to_pitch(token)
            if pitch is not None:
                pitches.add(pitch)
        
        if len(pitches) == 0:
            return 0.0
        
        # 다양성을 0-1 범위로 정규화 (최대 12개 음계)
        diversity = min(len(pitches) / 12.0, 1.0)
        return diversity
    
    def calculate_rhythm_structure(self, tokens: List[int]) -> float:
        """리듬 구조 보상 계산"""
        durations = []
        for token in tokens:
            duration = self.tokenizer.token_to_duration(token)
            if duration is not None:
                durations.append(duration)
        
        if len(durations) < 2:
            return 0.0
        
        # 리듬 패턴의 일관성 측정
        duration_variance = np.var(durations)
        
        # 적당한 변화가 있는 리듬을 선호 (너무 단조롭지도, 너무 불규칙하지도 않게)
        optimal_variance = 0.25
        rhythm_score = 1.0 - abs(duration_variance - optimal_variance) / optimal_variance
        
        return max(0.0, rhythm_score)
    
    def calculate_repetition_penalty(self, tokens: List[int]) -> float:
        """중복 회피 보상 계산"""
        if len(tokens) < 4:
            return 1.0
        
        # 연속된 동일 토큰 패널티
        repetition_count = 0
        for i in range(1, len(tokens)):
            if tokens[i] == tokens[i-1]:
                repetition_count += 1
        
        repetition_ratio = repetition_count / (len(tokens) - 1)
        penalty = 1.0 - repetition_ratio
        
        return max(0.0, penalty)
    
    def calculate_length_reward(self, tokens: List[int], target_length: int = 100) -> float:
        """적절한 길이 보상"""
        length = len(tokens)
        if length == 0:
            return 0.0
        
        # 목표 길이에 가까울수록 높은 보상
        length_diff = abs(length - target_length) / target_length
        length_reward = 1.0 - min(length_diff, 1.0)
        
        return length_reward
    
    def calculate_total_reward(self, tokens: List[int]) -> float:
        """전체 보상 계산"""
        # 각 보상 요소의 가중치
        weights = {
            'harmony': 0.3,
            'diversity': 0.2,
            'rhythm': 0.2,
            'repetition': 0.2,
            'length': 0.1
        }
        
        harmony_reward = self.calculate_harmony_consistency(tokens)
        diversity_reward = self.calculate_pitch_diversity(tokens)
        rhythm_reward = self.calculate_rhythm_structure(tokens)
        repetition_reward = self.calculate_repetition_penalty(tokens)
        length_reward = self.calculate_length_reward(tokens)
        
        total_reward = (
            weights['harmony'] * harmony_reward +
            weights['diversity'] * diversity_reward +
            weights['rhythm'] * rhythm_reward +
            weights['repetition'] * repetition_reward +
            weights['length'] * length_reward
        )
        
        return total_reward

class MusicGenerationEnv(gym.Env):
    """음악 생성을 위한 강화학습 환경"""
    
    def __init__(self, model: MusicGPT, tokenizer: MIDITokenizer, 
                 max_length: int = 200, device: torch.device = torch.device('cpu')):
        super().__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.reward_calculator = MusicRewardCalculator(tokenizer)
        
        # 액션 스페이스: 어휘 크기
        self.action_space = gym.spaces.Discrete(tokenizer.vocab_size)
        
        # 관찰 스페이스: 현재 시퀀스 (고정 길이)
        self.observation_space = gym.spaces.Box(
            low=0, high=tokenizer.vocab_size-1, 
            shape=(max_length,), dtype=np.int32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        
        # BOS 토큰으로 시작
        self.current_sequence = [self.tokenizer.special_tokens['[BOS]']]
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """현재 상태 관찰 반환"""
        # 현재 시퀀스를 고정 길이로 패딩
        obs = self.current_sequence + [0] * (self.max_length - len(self.current_sequence))
        return np.array(obs[:self.max_length], dtype=np.int32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        # 액션(토큰)을 시퀀스에 추가
        self.current_sequence.append(action)
        self.step_count += 1
        
        # 종료 조건 확인
        terminated = (
            action == self.tokenizer.special_tokens['[EOS]'] or
            len(self.current_sequence) >= self.max_length
        )
        
        # 보상 계산
        if terminated:
            reward = self.reward_calculator.calculate_total_reward(self.current_sequence)
        else:
            # 중간 보상 (작은 값)
            reward = 0.01
        
        truncated = False
        info = {
            'sequence_length': len(self.current_sequence),
            'final_reward': reward if terminated else 0.0
        }
        
        return self._get_observation(), reward, terminated, truncated, info

class PPOTrainer:
    """PPO 기반 강화학습 트레이너"""
    
    def __init__(self, model: MusicGPT, tokenizer: MIDITokenizer, config: Dict, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # PPO 하이퍼파라미터
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # 가치 함수 (간단한 선형 레이어)
        self.value_head = nn.Linear(model.d_model, 1).to(device)
        
        # 옵티마이저
        all_params = list(model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=config.get('learning_rate', 1e-5))
        
        # 로깅
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'rl_logs'))
        self.global_step = 0
        
        # 환경 생성
        self.env = MusicGenerationEnv(model, tokenizer, device=device)
    
    def get_action_and_value(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """상태에서 액션과 가치 계산"""
        # 모델 forward pass
        logits = self.model(states)
        
        # 마지막 토큰의 로짓만 사용
        last_logits = logits[:, -1, :]
        
        # 액션 분포
        action_dist = torch.distributions.Categorical(logits=last_logits)
        action = action_dist.sample()
        
        # 로그 확률
        log_prob = action_dist.log_prob(action)
        
        # 가치 계산 (모델의 마지막 히든 상태 사용)
        with torch.no_grad():
            hidden_states = self.model.token_embedding(states) * np.sqrt(self.model.d_model)
            hidden_states = self.model.position_encoding(hidden_states.transpose(0, 1)).transpose(0, 1)
            
            for layer in self.model.layers:
                hidden_states = layer(hidden_states)
            
            last_hidden = hidden_states[:, -1, :]
            value = self.value_head(last_hidden).squeeze(-1)
        
        return action, log_prob, value
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                   next_value: float, gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
        """Generalized Advantage Estimation 계산"""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[i + 1]
            
            delta = rewards[i] + gamma * next_val - values[i]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def collect_rollouts(self, num_steps: int) -> Dict:
        """롤아웃 데이터 수집"""
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        
        state, _ = self.env.reset()
        
        for step in range(num_steps):
            # 현재 상태를 텐서로 변환
            state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # 액션과 가치 계산
            with torch.no_grad():
                action, log_prob, value = self.get_action_and_value(state_tensor)
            
            # 환경에서 스텝 실행
            next_state, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated
            
            # 데이터 저장
            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
            rewards.append(reward)
            dones.append(done)
            
            state = next_state
            
            if done:
                state, _ = self.env.reset()
        
        # 마지막 상태의 가치 계산
        state_tensor = torch.tensor(state, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, next_value = self.get_action_and_value(state_tensor)
        
        # GAE 계산
        advantages, returns = self.compute_gae(rewards, values, next_value.item())
        
        return {
            'states': np.array(states),
            'actions': np.array(actions),
            'log_probs': np.array(log_probs),
            'values': np.array(values),
            'rewards': np.array(rewards),
            'advantages': np.array(advantages),
            'returns': np.array(returns),
            'dones': np.array(dones)
        }
    
    def update_policy(self, rollout_data: Dict, num_epochs: int = 4, batch_size: int = 64):
        """정책 업데이트"""
        states = torch.tensor(rollout_data['states'], dtype=torch.long).to(self.device)
        actions = torch.tensor(rollout_data['actions'], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor(rollout_data['log_probs']).to(self.device)
        advantages = torch.tensor(rollout_data['advantages']).to(self.device)
        returns = torch.tensor(rollout_data['returns']).to(self.device)
        
        # Advantage 정규화
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(states)
        
        for epoch in range(num_epochs):
            # 미니배치로 나누어 업데이트
            indices = torch.randperm(dataset_size)
            
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = min(start_idx + batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 현재 정책으로 로그 확률과 가치 계산
                logits = self.model(batch_states)
                last_logits = logits[:, -1, :]
                
                action_dist = torch.distributions.Categorical(logits=last_logits)
                new_log_probs = action_dist.log_prob(batch_actions)
                entropy = action_dist.entropy().mean()
                
                # 가치 계산
                hidden_states = self.model.token_embedding(batch_states) * np.sqrt(self.model.d_model)
                hidden_states = self.model.position_encoding(hidden_states.transpose(0, 1)).transpose(0, 1)
                
                for layer in self.model.layers:
                    hidden_states = layer(hidden_states)
                
                last_hidden = hidden_states[:, -1, :]
                values = self.value_head(last_hidden).squeeze(-1)
                
                # PPO 손실 계산
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = -entropy
                
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # 역전파
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.value_head.parameters()), 
                                             self.max_grad_norm)
                self.optimizer.step()
                
                # 로깅
                if self.global_step % 100 == 0:
                    self.writer.add_scalar('rl/policy_loss', policy_loss.item(), self.global_step)
                    self.writer.add_scalar('rl/value_loss', value_loss.item(), self.global_step)
                    self.writer.add_scalar('rl/entropy', entropy.item(), self.global_step)
                    self.writer.add_scalar('rl/total_loss', total_loss.item(), self.global_step)
                
                self.global_step += 1
    
    def train(self, num_iterations: int, rollout_steps: int = 2048):
        """강화학습 훈련"""
        print(f"Starting RL training for {num_iterations} iterations")
        
        for iteration in tqdm(range(num_iterations), desc="RL Training"):
            # 롤아웃 수집
            rollout_data = self.collect_rollouts(rollout_steps)
            
            # 정책 업데이트
            self.update_policy(rollout_data)
            
            # 통계 로깅
            avg_reward = np.mean(rollout_data['rewards'])
            avg_episode_length = np.mean([len(rollout_data['rewards'])])
            
            self.writer.add_scalar('rl/avg_reward', avg_reward, iteration)
            self.writer.add_scalar('rl/avg_episode_length', avg_episode_length, iteration)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: Avg Reward = {avg_reward:.4f}")
            
            # 체크포인트 저장
            if iteration % 100 == 0:
                checkpoint_path = os.path.join(self.config.get('checkpoint_dir', 'rl_checkpoints'), 
                                             f'rl_checkpoint_{iteration}.pt')
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'value_head_state_dict': self.value_head.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': iteration
                }, checkpoint_path)
        
        print("RL training completed!")
        self.writer.close()

def get_rl_config() -> Dict:
    """기본 강화학습 설정 반환"""
    return {
        'learning_rate': 1e-5,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'checkpoint_dir': 'rl_checkpoints',
        'log_dir': 'rl_logs'
    }

if __name__ == "__main__":
    # 사용 예시
    from .model import MusicGPT, MusicGPTConfig
    from .data_preprocessing import MIDITokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 토크나이저와 모델 생성
    tokenizer = MIDITokenizer()
    model_config = MusicGPTConfig(vocab_size=tokenizer.vocab_size)
    model = MusicGPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers
    )
    
    # RL 트레이너 생성
    rl_config = get_rl_config()
    trainer = PPOTrainer(model, tokenizer, rl_config, device)
    
    print("RL trainer created successfully")