import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageTk, Image
from bingo_artist import *
from functools import partial
from tkinter import filedialog, messagebox
from bingo_table_maker import bomb_explode

MODE_FLIP = 0
MODE_INIT = 1
MODE_BOMB = 2

INFO_LOADING = '로딩중입니다. 잠시 기다려주세요.'
INFO_INIT_TWO = '첫 두 칸을 뒤집어주세요'
INFO_FLIP = '수동 뒤집기 모드'
INFO_BOMB = '폭탄 모드'

WAR_IMP = '무력화 불가!!'
WAR_UNK = '불가능한 상태입니다. \n(이난나 사용하셨나요?)'


class Console():
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('KukuBingo')
        self.root.geometry('1300x700')
        self.root.resizable(False,False)

        self.bingo_artist = BingoArtist()
        self.reset(update=False)

        self.label_main_bingo= ttk.Label(self.root)
        self.label_main_bingo.place(x=0,y=0)

        self.bingo_buttons = []
        for x in range(5):
            row_buttons = []
            for y in range(5):
                new_button = ttk.Button(self.root,width=5,text=f'{x+1},{y+1}',
                        command=partial(self.bingo_button_callback,x,y))
                new_button.place(x=MIDDLE-20-SPACE*x+SPACE*y,
                                 y=SPACE*(x+y+1))
                row_buttons.append(new_button)
            self.bingo_buttons.append(row_buttons)

        self.mode_variable = tk.StringVar(value='')
        self.label_current_mode = ttk.Label(
            self.root,
            textvariable=self.mode_variable,
            font=(None,13,'bold'),
            background="white",
            foreground="red"
        )
        self.label_current_mode.place(x=20,y=20)

        self.step_variable = tk.StringVar(value='')
        self.label_step = ttk.Label(
            self.root,
            textvariable=self.step_variable,
            font=(None, 13, 'bold'),
            background='white'
        )
        self.label_step.place(x=20,y=50)

        self.weak_variable = tk.StringVar(value='')
        self.label_weak = ttk.Label(
            self.root,
            textvariable=self.weak_variable,
            font=(None, 13, 'bold'),
            background='white'
        )
        self.label_weak.place(x=20, y=80)

        self.warning_variable = tk.StringVar(value='')
        self.label_warning = ttk.Label(
            self.root,
            textvariable=self.warning_variable,
            font=(None, 15, 'bold'),
            background='white',
            foreground='red'
        )
        self.label_warning.place(x=MIDDLE+100, y=20)

        self.button_flip_mode = ttk.Button(
            self.root,
            text='수동모드',
            command=self.button_flip_callback,
        )
        self.button_flip_mode.place(x=SIZE+30, y=50)

        self.button_bomb_mode = ttk.Button(
            self.root,
            text='폭탄모드',
            command=self.button_bomb_callback
        )
        self.button_bomb_mode.place(x=SIZE+30,y=100)

        self.button_reset = ttk.Button(
            self.root,
            text='리셋',
            command=self.button_reset_callback
        )
        self.button_reset.place(x=SIZE+200,y=50)

        self.button_cancel = ttk.Button(
            self.root,
            text='하나취소',
            command=self.button_cancel_callback
        )
        self.button_cancel.place(x=SIZE+200, y=100)
        
        self.label_future_1 = ttk.Label(self.root)
        self.label_future_1.place(x=SIZE+10, y=MIDDLE)
        self.label_future_2 = ttk.Label(self.root)
        self.label_future_2.place(x=SIZE+MIDDLE+10, y=MIDDLE)



    def reset(self, update=True):
        self.click_history = []
        if update:
            self.update()
        

    def update(self):
        self.bingo_board = np.zeros((5,5),dtype=bool)
        self.future_img1 = ImageTk.PhotoImage(Image.fromarray(
                    np.zeros((MIDDLE,MIDDLE,3),np.uint8)))
        self.future_img2 = ImageTk.PhotoImage(Image.fromarray(
                    np.zeros((MIDDLE,MIDDLE,3),np.uint8)))
        self.step = 0
        for r in self.bingo_buttons:
            for bb in r:
                bb.state(['!disabled'])

        if len(self.click_history)<2:
            self.mode = MODE_INIT
            self.mode_variable.set(INFO_INIT_TWO)
            self.button_bomb_mode.state(['disabled'])
            self.button_flip_mode.state(['disabled'])
            for x, y, _ in self.click_history:
                self.bingo_buttons[x][y].state(['disabled'])
        else:
            self.button_bomb_mode.state(['!disabled'])
            self.button_flip_mode.state(['!disabled'])
            if len(self.click_history)==2 and self.mode==MODE_INIT:
                self.mode = MODE_BOMB

        for x, y, m in self.click_history:
            if m in [MODE_INIT, MODE_FLIP]:
                self.bingo_board[x,y] = not self.bingo_board[x,y]
            elif m == MODE_BOMB:
                self.bingo_board = bomb_explode(
                    self.bingo_board,(x,y)
                )
                self.step += 1

        self.warning_variable.set('')
        self.weak_variable.set('')

        table_idx = np.append(self.bingo_board.reshape(-1).astype(int),
                              self.step%3)
        table_idx = tuple(table_idx)
        if self.table_filled[table_idx]:
            rec_x, rec_y, max_w, _, _ = self.table[table_idx]
            rec_pos = (int(rec_x),int(rec_y))
            self.weak_variable.set(f'앞으로 {int(max_w)}무력 가능')
            if max_w ==0:
                self.warning_variable.set(WAR_IMP)
                rec_pos = None
                
            else:
                next_board = bomb_explode(self.bingo_board,rec_pos)
                next_idx = np.append(next_board.reshape(-1).astype(int),
                                (self.step+1)%3)
                next_idx = tuple(next_idx)
                next_x, next_y, next_w, _, _ = self.table[next_idx]
                next_pos = (int(next_x),int(next_y))
                self.future_img1 = ImageTk.PhotoImage(Image.fromarray(
                    self.bingo_artist.draw_board(next_board,next_pos,small=True)
                ))
                if next_w>0:
                    next_board = bomb_explode(next_board, next_pos)
                    next_idx = np.append(next_board.reshape(-1).astype(int),
                                (self.step+2)%3)
                    next_idx = tuple(next_idx)
                    next_x, next_y, _,_,_ = self.table[next_idx]
                    next_pos = (int(next_x),int(next_y))
                    self.future_img2 = ImageTk.PhotoImage(Image.fromarray(
                        self.bingo_artist.draw_board(next_board,next_pos,small=True)
                    ))
        else:
            rec_pos = None
            
            if len(self.click_history)>2:
                self.warning_variable.set(WAR_UNK)

        if not self.mode==MODE_INIT:
            text = f'{self.step+1}번째 폭탄'
            if self.step%3==2:
                text += '(이번에 무력화)'
            self.step_variable.set(text)
        else:
            self.step_variable.set('')

        if self.mode == MODE_FLIP:
            self.mode_variable.set(INFO_FLIP)
        elif self.mode == MODE_BOMB:
            self.mode_variable.set(INFO_BOMB)

        self.bingo_img = ImageTk.PhotoImage(Image.fromarray(
                self.bingo_artist.draw_board(self.bingo_board,rec_pos)))
        self.label_main_bingo.configure(
            image=self.bingo_img
        )
        self.label_future_1.configure(image=self.future_img1)
        self.label_future_2.configure(image=self.future_img2)

    def bingo_button_callback(self,x,y):
        self.click_history.append((x,y,self.mode))
        self.update()

    def button_flip_callback(self):
        self.mode = MODE_FLIP
        self.update()

    def button_bomb_callback(self):
        self.mode = MODE_BOMB
        self.update()


    def button_reset_callback(self):
        self.reset()

    def button_cancel_callback(self):
        if len(self.click_history)>0:
            self.click_history.pop()
            self.update()

        
    def run(self):
        self.mode_variable.set(INFO_LOADING)
        dirname = filedialog.askopenfilename(filetypes=[('넘파이 압축 파일','*.npz')])
        try:
            loaded_file = np.load(dirname.encode('unicode_escape'))
            self.table = loaded_file['table']
            self.table_filled = loaded_file['filled']
        except(IOError, ValueError):
            messagebox.showinfo(message=('테이블 로드에 실패했습니다.\n'
                                '경로에 한글이 있는지 확인해주세요.'))
            return
        self.mode_variable.set(INFO_INIT_TWO)
        self.update()
        self.root.mainloop()

if __name__ == '__main__':
    Console().run()