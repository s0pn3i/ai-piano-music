#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
개선된 MIDI 파일 선택 및 재생 프로그램
다양한 백엔드를 지원하여 재생 호환성을 높였습니다.
"""

import os
import sys
import glob
import time
import threading
import subprocess
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
        self.current_file = None
        self.is_playing = False
        self.midi_files = []
        self.available_backends = self.check_available_backends()
        
    def check_available_backends(self):
        """사용 가능한 MIDI 재생 백엔드 확인"""
        backends = []
        
        # pygame 테스트
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            backends.append('pygame')
            print("✓ pygame 백엔드 사용 가능")
        except Exception as e:
            print(f"✗ pygame 백엔드 오류: {e}")
        
        # timidity 확인
        try:
            result = subprocess.run(['which', 'timidity'], capture_output=True, text=True)
            if result.returncode == 0:
                backends.append('timidity')
                print("✓ timidity 백엔드 사용 가능")
        except:
            print("✗ timidity 백엔드 없음")
        
        # fluidsynth 확인
        try:
            result = subprocess.run(['which', 'fluidsynth'], capture_output=True, text=True)
            if result.returncode == 0:
                backends.append('fluidsynth')
                print("✓ fluidsynth 백엔드 사용 가능")
        except:
            print("✗ fluidsynth 백엔드 없음")
        
        # aplay 확인 (MIDI를 WAV로 변환 후 재생)
        try:
            result = subprocess.run(['which', 'aplay'], capture_output=True, text=True)
            if result.returncode == 0:
                backends.append('aplay')
                print("✓ aplay 백엔드 사용 가능")
        except:
            print("✗ aplay 백엔드 없음")
        
        if not backends:
            print("⚠️  사용 가능한 MIDI 재생 백엔드가 없습니다.")
            print("   시각화 기능만 사용할 수 있습니다.")
        
        return backends
    
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
        print(f"{len(self.midi_files) + 3:2d}. MIDI → WAV 변환")
        print(f"{len(self.midi_files) + 4:2d}. 백엔드 상태 확인")
        print(f"{len(self.midi_files) + 5:2d}. 종료")
        return True
    
    def play_with_pygame(self, file_path):
        """pygame으로 MIDI 재생"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            return True
        except Exception as e:
            print(f"pygame 재생 실패: {e}")
            return False
    
    def play_with_timidity(self, file_path):
        """timidity로 MIDI 재생"""
        try:
            result = subprocess.run(['timidity', file_path], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            print(f"timidity 재생 실패: {e}")
            return False
    
    def play_with_fluidsynth(self, file_path):
        """fluidsynth로 MIDI 재생"""
        try:
            # 사운드폰트 파일 찾기
            soundfont_paths = [
                '/usr/share/sounds/sf2/FluidR3_GM.sf2',
                '/usr/share/sounds/sf2/default.sf2',
                '/usr/share/soundfonts/default.sf2'
            ]
            
            soundfont = None
            for sf_path in soundfont_paths:
                if os.path.exists(sf_path):
                    soundfont = sf_path
                    break
            
            if not soundfont:
                print("사운드폰트 파일을 찾을 수 없습니다.")
                return False
            
            result = subprocess.run([
                'fluidsynth', '-a', 'alsa', '-g', '0.5', 
                soundfont, file_path
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception as e:
            print(f"fluidsynth 재생 실패: {e}")
            return False
    
    def convert_to_wav(self, file_path):
        """MIDI를 WAV로 변환"""
        try:
            midi_data = pretty_midi.PrettyMIDI(file_path)
            audio_data = midi_data.synthesize(fs=44100)
            
            # WAV 파일로 저장
            output_path = file_path.replace('.mid', '.wav')
            
            import scipy.io.wavfile as wavfile
            # 오디오 데이터 정규화
            audio_data = audio_data / np.max(np.abs(audio_data))
            audio_data = (audio_data * 32767).astype(np.int16)
            
            wavfile.write(output_path, 44100, audio_data)
            print(f"WAV 파일 생성: {output_path}")
            
            # aplay로 재생
            if 'aplay' in self.available_backends:
                subprocess.run(['aplay', output_path])
            
            return True
        except Exception as e:
            print(f"WAV 변환 실패: {e}")
            return False
    
    def play_midi(self, file_path):
        """사용 가능한 백엔드로 MIDI 재생 시도"""
        print(f"\n재생 중: {os.path.basename(file_path)}")
        
        # 백엔드 우선순위: timidity > fluidsynth > pygame > wav변환
        backends_to_try = ['timidity', 'fluidsynth', 'pygame']
        
        for backend in backends_to_try:
            if backend in self.available_backends:
                print(f"백엔드 시도: {backend}")
                
                if backend == 'pygame':
                    success = self.play_with_pygame(file_path)
                elif backend == 'timidity':
                    success = self.play_with_timidity(file_path)
                elif backend == 'fluidsynth':
                    success = self.play_with_fluidsynth(file_path)
                
                if success:
                    print(f"✓ {backend}로 재생 성공!")
                    return True
                else:
                    print(f"✗ {backend} 재생 실패")
        
        # 모든 백엔드 실패 시 WAV 변환 시도
        print("모든 MIDI 백엔드 실패. WAV 변환을 시도합니다...")
        return self.convert_to_wav(file_path)
    
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
    
    def show_backend_status(self):
        """백엔드 상태 표시"""
        print("\n=== MIDI 재생 백엔드 상태 ===")
        
        all_backends = ['pygame', 'timidity', 'fluidsynth', 'aplay']
        
        for backend in all_backends:
            status = "✓ 사용 가능" if backend in self.available_backends else "✗ 사용 불가"
            print(f"{backend:12s}: {status}")
        
        if not self.available_backends:
            print("\n⚠️  해결 방법:")
            print("1. TiMidity++ 설치: sudo apt install timidity freepats")
            print("2. FluidSynth 설치: sudo apt install fluidsynth fluid-soundfont-gm")
            print("3. 또는 WAV 변환 기능을 사용하세요.")
    
    def get_user_choice(self):
        """사용자 선택 입력 받기"""
        try:
            choice = input("\n선택하세요 (번호 입력): ").strip()
            return int(choice)
        except ValueError:
            return -1
    
    def run(self):
        """메인 실행 루프"""
        print("🎵 개선된 MIDI 파일 플레이어 🎵")
        print("생성된 AI 음악을 선택하여 재생하세요!")
        print(f"사용 가능한 백엔드: {', '.join(self.available_backends) if self.available_backends else '없음'}")
        
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
                
                if self.available_backends:
                    print("\n재생 옵션:")
                    print("1. 재생")
                    print("2. 시각화")
                    print("3. 재생 + 시각화")
                    print("4. WAV 변환")
                    
                    option = self.get_user_choice()
                    
                    if option == 1:
                        self.play_midi(selected_file)
                    elif option == 2:
                        self.visualize_midi(selected_file)
                    elif option == 3:
                        play_thread = threading.Thread(target=self.play_midi, args=(selected_file,))
                        play_thread.start()
                        time.sleep(1)
                        self.visualize_midi(selected_file)
                        play_thread.join()
                    elif option == 4:
                        self.convert_to_wav(selected_file)
                    else:
                        print("올바른 옵션을 선택하세요.")
                else:
                    print("\n재생 백엔드가 없습니다. 시각화만 가능합니다.")
                    self.visualize_midi(selected_file)
            
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
                if self.available_backends:
                    print("\n=== 모든 파일 연속 재생 ===")
                    for i, file_path in enumerate(self.midi_files, 1):
                        print(f"\n[{i}/{len(self.midi_files)}] 재생 중...")
                        self.play_midi(file_path)
                        if i < len(self.midi_files):
                            time.sleep(2)
                else:
                    print("재생 백엔드가 없습니다.")
            
            # WAV 변환
            elif choice == len(self.midi_files) + 3:
                print("\n=== MIDI → WAV 변환 ===")
                for file_path in self.midi_files:
                    print(f"변환 중: {os.path.basename(file_path)}")
                    self.convert_to_wav(file_path)
            
            # 백엔드 상태
            elif choice == len(self.midi_files) + 4:
                self.show_backend_status()
            
            # 종료
            elif choice == len(self.midi_files) + 5:
                print("\n프로그램을 종료합니다. 안녕히 가세요! 👋")
                break
            
            else:
                print("올바른 번호를 선택하세요.")
            
            if choice != len(self.midi_files) + 5:
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
        try:
            pygame.mixer.quit()
        except:
            pass

if __name__ == "__main__":
    main()