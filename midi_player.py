#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIDI 파일 선택 및 재생 프로그램
생성된 MIDI 파일들을 목록에서 선택하여 재생할 수 있습니다.
"""

import os
import sys
import glob
import time
import threading
from pathlib import Path

try:
    import pygame
except ImportError:
    print("pygame이 설치되지 않았습니다. 설치 중...")
    os.system("pip install pygame")
    import pygame

try:
    import pretty_midi
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("필요한 라이브러리가 설치되지 않았습니다.")
    print("다음 명령으로 설치하세요: pip install pretty_midi matplotlib numpy")
    sys.exit(1)

class MIDIPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.current_file = None
        self.is_playing = False
        self.midi_files = []
        
    def find_midi_files(self, directory="demo_output/generated_music"):
        """MIDI 파일들을 찾아서 목록으로 반환"""
        if not os.path.exists(directory):
            print(f"디렉토리를 찾을 수 없습니다: {directory}")
            return []
            
        midi_patterns = ["*.mid", "*.midi"]
        midi_files = []
        
        for pattern in midi_patterns:
            midi_files.extend(glob.glob(os.path.join(directory, pattern)))
            
        return sorted(midi_files)
    
    def display_file_list(self):
        """MIDI 파일 목록을 표시"""
        self.midi_files = self.find_midi_files()
        
        if not self.midi_files:
            print("MIDI 파일을 찾을 수 없습니다.")
            print("demo.py를 먼저 실행하여 MIDI 파일을 생성하세요.")
            return False
            
        print("\n=== 사용 가능한 MIDI 파일 ===")
        for i, file_path in enumerate(self.midi_files, 1):
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            print(f"{i:2d}. {filename} ({file_size} bytes)")
        
        print(f"{len(self.midi_files) + 1:2d}. 피아노 롤 시각화")
        print(f"{len(self.midi_files) + 2:2d}. 모든 파일 연속 재생")
        print(f"{len(self.midi_files) + 3:2d}. 종료")
        return True
    
    def play_midi(self, file_path):
        """MIDI 파일 재생"""
        try:
            print(f"\n재생 중: {os.path.basename(file_path)}")
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            self.current_file = file_path
            self.is_playing = True
            
            # 재생 상태 모니터링
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            self.is_playing = False
            print("재생 완료!")
            
        except pygame.error as e:
            print(f"재생 오류: {e}")
            print("다른 MIDI 플레이어를 사용해보세요.")
    
    def stop_playback(self):
        """재생 중지"""
        if self.is_playing:
            pygame.mixer.music.stop()
            self.is_playing = False
            print("재생이 중지되었습니다.")
    
    def visualize_midi(self, file_path):
        """MIDI 파일을 피아노 롤로 시각화"""
        try:
            print(f"\n시각화 중: {os.path.basename(file_path)}")
            midi_data = pretty_midi.PrettyMIDI(file_path)
            
            plt.figure(figsize=(15, 8))
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, instrument in enumerate(midi_data.instruments):
                color = colors[i % len(colors)]
                for note in instrument.notes:
                    plt.plot([note.start, note.end], [note.pitch, note.pitch], 
                            color=color, linewidth=2, alpha=0.7)
            
            plt.xlabel('시간 (초)', fontsize=12)
            plt.ylabel('음높이 (MIDI 노트 번호)', fontsize=12)
            plt.title(f'피아노 롤 시각화: {os.path.basename(file_path)}', fontsize=14)
            plt.grid(True, alpha=0.3)
            
            # 음높이 레이블 추가
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            yticks = range(21, 109, 12)  # C1부터 C8까지
            yticklabels = [f"{note_names[0]}{(tick-12)//12}" for tick in yticks]
            plt.yticks(yticks, yticklabels)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"시각화 오류: {e}")
    
    def play_all_files(self):
        """모든 MIDI 파일을 연속으로 재생"""
        print("\n=== 모든 파일 연속 재생 ===")
        for i, file_path in enumerate(self.midi_files, 1):
            print(f"\n[{i}/{len(self.midi_files)}] 재생 중...")
            self.play_midi(file_path)
            
            if i < len(self.midi_files):
                print("다음 파일로 이동... (3초 대기)")
                time.sleep(3)
        
        print("\n모든 파일 재생 완료!")
    
    def get_user_choice(self):
        """사용자 선택 입력 받기"""
        try:
            choice = input("\n선택하세요 (번호 입력): ").strip()
            return int(choice)
        except ValueError:
            return -1
    
    def run(self):
        """메인 실행 루프"""
        print("🎵 MIDI 파일 플레이어 🎵")
        print("생성된 AI 음악을 선택하여 재생하세요!")
        
        while True:
            if not self.display_file_list():
                break
                
            choice = self.get_user_choice()
            
            if choice == -1:
                print("올바른 번호를 입력하세요.")
                continue
            
            # 개별 파일 재생
            if 1 <= choice <= len(self.midi_files):
                selected_file = self.midi_files[choice - 1]
                
                print("\n재생 옵션:")
                print("1. 재생")
                print("2. 시각화")
                print("3. 재생 + 시각화")
                
                option = self.get_user_choice()
                
                if option == 1:
                    self.play_midi(selected_file)
                elif option == 2:
                    self.visualize_midi(selected_file)
                elif option == 3:
                    # 별도 스레드에서 재생
                    play_thread = threading.Thread(target=self.play_midi, args=(selected_file,))
                    play_thread.start()
                    time.sleep(1)  # 재생 시작 대기
                    self.visualize_midi(selected_file)
                    play_thread.join()
                else:
                    print("올바른 옵션을 선택하세요.")
            
            # 시각화 메뉴
            elif choice == len(self.midi_files) + 1:
                print("\n시각화할 파일을 선택하세요:")
                for i, file_path in enumerate(self.midi_files, 1):
                    print(f"{i}. {os.path.basename(file_path)}")
                
                viz_choice = self.get_user_choice()
                if 1 <= viz_choice <= len(self.midi_files):
                    self.visualize_midi(self.midi_files[viz_choice - 1])
            
            # 연속 재생
            elif choice == len(self.midi_files) + 2:
                self.play_all_files()
            
            # 종료
            elif choice == len(self.midi_files) + 3:
                print("\n프로그램을 종료합니다. 안녕히 가세요! 👋")
                break
            
            else:
                print("올바른 번호를 선택하세요.")
            
            # 계속 진행 여부 확인
            if choice != len(self.midi_files) + 3:
                input("\nEnter 키를 눌러 계속하세요...")
                print("\n" + "="*50)

def main():
    """메인 함수"""
    try:
        player = MIDIPlayer()
        player.run()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        print("프로그램을 다시 시작해보세요.")
    finally:
        pygame.mixer.quit()

if __name__ == "__main__":
    main()