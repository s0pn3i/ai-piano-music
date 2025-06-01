import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import json
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np
from .model import MusicGPT, MusicGPTConfig
from .data_preprocessing import create_pytorch_dataset

class MusicGPTTrainer:
    """MusicGPT 모델 훈련을 위한 클래스"""
    
    def __init__(self, model: MusicGPT, config: Dict, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['max_epochs'],
            eta_min=config['learning_rate'] * 0.1
        )
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD 토큰 무시
        
        # 텐서보드 로거
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # 훈련 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 디바이스로 이동
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            
            # 손실 계산
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            
            # 옵티마이저 스텝
            self.optimizer.step()
            
            # 통계 업데이트
            batch_loss = loss.item()
            batch_tokens = (labels != 0).sum().item()  # PAD 토큰 제외
            
            total_loss += batch_loss * batch_tokens
            total_tokens += batch_tokens
            
            # 로깅
            if self.global_step % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/loss', batch_loss, self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # 진행률 표시 업데이트
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                batch_tokens = (labels != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity
        }
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = filepath.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_filepath)
    
    def load_checkpoint(self, filepath: str):
        """체크포인트 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """전체 훈련 루프"""
        print(f"Starting training for {self.config['max_epochs']} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config['max_epochs']):
            self.current_epoch = epoch
            
            # 훈련
            train_metrics = self.train_epoch(train_loader)
            
            # 검증
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # 학습률 스케줄러 업데이트
            self.scheduler.step()
            
            # 로깅
            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/train_perplexity', train_metrics['perplexity'], epoch)
            
            if val_metrics:
                self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('epoch/val_perplexity', val_metrics['perplexity'], epoch)
            
            # 결과 출력
            print(f"Epoch {epoch + 1}/{self.config['max_epochs']}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Perplexity: {train_metrics['perplexity']:.2f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            
            # 체크포인트 저장
            if epoch % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(self.config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pt')
                self.save_checkpoint(checkpoint_path)
            
            # 최고 성능 모델 저장
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                best_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                self.save_checkpoint(best_path, is_best=True)
                print(f"  New best model saved with loss: {self.best_loss:.4f}")
        
        print("Training completed!")
        self.writer.close()

def create_data_loaders(train_data_file: str, val_data_file: Optional[str] = None,
                       batch_size: int = 32, max_length: int = 512,
                       num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
    """데이터 로더 생성"""
    
    # 훈련 데이터셋
    train_dataset = create_pytorch_dataset(train_data_file, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 검증 데이터셋
    val_loader = None
    if val_data_file:
        val_dataset = create_pytorch_dataset(val_data_file, max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader

def get_default_config() -> Dict:
    """기본 훈련 설정 반환"""
    return {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'max_epochs': 100,
        'max_grad_norm': 1.0,
        'log_interval': 100,
        'save_interval': 10,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    }

if __name__ == "__main__":
    # 사용 예시
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 모델 설정
    model_config = MusicGPTConfig(vocab_size=1000)
    model = MusicGPT(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_heads=model_config.n_heads,
        n_layers=model_config.n_layers
    )
    
    # 훈련 설정
    train_config = get_default_config()
    
    # 트레이너 생성
    trainer = MusicGPTTrainer(model, train_config, device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")