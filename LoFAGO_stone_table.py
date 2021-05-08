import tkinter as tk
from tkinter import ttk
from functools import partial
from pathlib import Path
import numpy as np
from tkinter import filedialog, messagebox
from stone_table_maker import get_q


EMPTY = "◇"
SUCC = "◆"
FAIL = "✕"
MAX_PROB = 75
MIN_PROB = 25
DEFAULT_TARGET = [7,6,2]
EXPLANATION=(
    '테이블 파일이 없어도 실행이 가능하나,\n'
    '매번 계산을 해야해서 첫 클릭 시\n'
    '렉이 걸릴 수 있습니다.'
)

class Console():
    def __init__(self):

        self.root = tk.Tk()
        self.root.title('AbilityStone')
        self.mainframe = ttk.Frame(self.root, padding='3 3 12 12')
        self.mainframe.grid(column=0, row=0, sticky=(tk.NSEW))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.resizable(False,False)

        self.button_right = True

        self.default_target = DEFAULT_TARGET
        self.reset(initial=True)

        self.click_history = []
        
        self.target_string_var = \
            tk.StringVar(value=self.target_string(self.target))
        self.save_string_var = tk.StringVar(
            value=self.save_string
        )

        self.a1_string_var = tk.StringVar(value=self.a1_string)
        self.a2_string_var = tk.StringVar(value=self.a2_string)
        self.b_string_var = tk.StringVar(value=self.b_string)

        self.prob = MAX_PROB
        self.prob_string_var = tk.StringVar(value=self.prob_string)
        
        self.pred_a1_string_var = tk.StringVar(value='--%')
        self.pred_a2_string_var = tk.StringVar(value='--%')
        self.pred_b_string_var = tk.StringVar(value='--%')

        self.explanation_string_var = tk.StringVar(
            value=" "
        )

        self.label_prob = ttk.Label(self.mainframe,
                            textvariable=self.prob_string_var,
                            padding='5 5 5 5',
                            font=(None,16,'bold'))
        self.label_prob.grid(column=0, row=0)
        self.label_chance = ttk.Label(
            self.mainframe,
            text='칸 수: '
        )
        self.label_chance.grid(column=1, row=0,sticky=[tk.E])
        self.combo_chance = ttk.Combobox(
            self.mainframe,
            values=[
                10,9,8,7,6
            ],
            state='readonly',
            width=10,
        )
        self.combo_chance.current(0)
        self.combo_chance.grid(column=2, row=0)

        self.button_cancel = ttk.Button(
            self.mainframe,
            text='하나 취소',
            command=self.cancel_one
        )
        self.button_cancel.grid(column=3, row=0)
        self.button_reset = ttk.Button(
            self.mainframe,
            text='리셋',
            command=self.reset
        )
        self.button_reset.grid(column=4, row=0)

        self.label_a1 = ttk.Label(
            self.mainframe,
            textvariable=self.a1_string_var,
            foreground='#00f',
            font=12
        )
        self.label_a1.grid(column=0, columnspan=3, row=1,padx=10, pady=10)
        self.button_a1_s = ttk.Button(
            self.mainframe,
            text='성공',
            command=partial(self.ability_button_callback,0,True),
        )
        self.button_a1_s.grid(column=3, row=1)
        self.button_a1_f = ttk.Button(
            self.mainframe,
            text='실패',
            command=partial(self.ability_button_callback,0,False),
        )
        self.button_a1_f.grid(column=4, row=1)

        self.label_a2 = ttk.Label(
            self.mainframe,
            textvariable=self.a2_string_var,
            foreground='#00f',
            font=12
        )
        self.label_a2.grid(column=0, columnspan=3, row=2,padx=10, pady=10)
        self.button_a2_s = ttk.Button(
            self.mainframe,
            text='성공',
            command=partial(self.ability_button_callback,1,True)
        )
        self.button_a2_s.grid(column=3, row=2)
        self.button_a2_f = ttk.Button(
            self.mainframe,
            text='실패',
            command=partial(self.ability_button_callback,1,False),
        )
        self.button_a2_f.grid(column=4, row=2)

        self.label_b = ttk.Label(
            self.mainframe,
            textvariable=self.b_string_var,
            foreground='#f00',
            font=12
        )
        self.label_b.grid(column=0, columnspan=3, row=3,padx=10, pady=10)
        self.button_b_s = ttk.Button(
            self.mainframe,
            text='성공',
            command=partial(self.ability_button_callback,2,True),
        )
        self.button_b_s.grid(column=3, row=3)
        self.button_b_f = ttk.Button(
            self.mainframe,
            text='실패',
            command=partial(self.ability_button_callback,2,False),
        )
        self.button_b_f.grid(column=4, row=3)

        self.label_pred_title = ttk.Label(
            self.mainframe,
            text='목표 달성 확률'
        )
        self.label_pred_title.grid(
            column=5, row=0, padx=5
        )
        self.label_pred_a1 = ttk.Label(
            self.mainframe,
            textvariable=self.pred_a1_string_var
        )
        self.label_pred_a1.grid(
            column=5, row=1, padx=5
        )

        self.label_pred_a2 = ttk.Label(
            self.mainframe,
            textvariable=self.pred_a2_string_var
        )
        self.label_pred_a2.grid(
            column=5, row=2, padx=5
        )
        self.label_pred_b = ttk.Label(
            self.mainframe,
            textvariable=self.pred_b_string_var
        )
        self.label_pred_b.grid(
            column=5, row=3, padx=5
        )

        self.label_target_title = ttk.Label(
            self.mainframe,
            text='목표 : ',
            font=(None,14,'bold')
        )
        self.label_target_title.grid(
            column=0, row=4
        )
        self.label_target_target = ttk.Label(
            self.mainframe,
            textvariable=self.target_string_var,
            font=(None,14,'bold')
        )
        self.label_target_target.grid(
            column=1, row=4,columnspan=2
        )

        self.button_save = ttk.Button(
            self.mainframe,
            text='목표 저장',
            command=self.save_button_callback,
        )
        self.button_save.grid(column=3, row=4)
        self.label_save = ttk.Label(
            self.mainframe,
            textvariable=self.save_string_var,
            font=12
        )
        self.label_save.grid(column=4, row=4)

        self.button_flip = ttk.Button(
            self.mainframe,
            text='좌우 바꾸기',
            command=self.flip_button_callback
        )
        self.button_flip.grid(column=5, row=4)

        self.frame_target = ttk.Frame(self.mainframe)
        self.frame_target.grid(column=0, columnspan=3, row=5, sticky=tk.NSEW)
        self.label_target_a1_title = ttk.Label(
            self.frame_target,
            text='좋은각인 1'
        )
        self.label_target_a2_title = ttk.Label(
            self.frame_target,
            text='좋은각인 2'
        )
        self.label_target_b_title = ttk.Label(
            self.frame_target,
            text='나쁜각인'
        )
        self.label_target_a1_title.grid(column=0, row=0, padx=10)
        self.label_target_a2_title.grid(column=1, row=0, padx=10)
        self.label_target_b_title.grid(column=2, row=0, padx=10)

        self.button_target_a1_list = []
        for i in range(11):
            self.button_target_a1_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 0, 10-i)
                )
            )
            self.button_target_a1_list[-1].grid(column=0, row=i+1)


        self.button_target_a2_list = []
        for i in range(11):
            self.button_target_a2_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 1, 10-i)
                )
            )
            self.button_target_a2_list[-1].grid(column=1, row=i+1, padx=10)

        self.button_target_b_list = []
        for i in range(11):
            self.button_target_b_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 2, 10-i)
                )
            )
            self.button_target_b_list[-1].grid(column=2, row=i+1)

        self.label_explanation = ttk.Label(
            self.mainframe,
            textvariable=self.explanation_string_var,
            font=13
        )
        self.label_explanation.grid(column=3, row=5, columnspan=3)


    @property
    def a1_string(self):
        return ' '.join(self.a1)
    @property
    def a2_string(self):
        return ' '.join(self.a2)
    @property
    def b_string(self):
        return ' '.join(self.b)

    @property
    def prob_string(self):
        return f'{self.prob}%'
    
    @property
    def save_string(self):
        return (f'{self.default_target[0]}'
                f'/{self.default_target[1]}'
                f'/{self.default_target[2]}')

    @property
    def prob_idx(self):
        return (self.prob-25)//10

    def target_string(self, target_left):
        postfixes = []
        for i in range(2):
            if target_left[i] <= 0:
                postfixes.append('(달성)')
            else:
                postfixes.append(f'({target_left[i]} 남음)')
        postfixes.append(f'(여유 {max(0,target_left[2])})')
        return (f'{self.target[0]} ' + postfixes[0] +
                f' / {self.target[1]} ' + postfixes[1]+
                f' / {self.target[2]} ' + postfixes[2])

    def prob_down(self):
        self.prob = max(self.prob-10, MIN_PROB)
    
    def prob_up(self):
        self.prob = min(self.prob+10, MAX_PROB)

    def target_button_callback(self, ability, N):
        self.target[ability] = N
        self.update()

    def ability_button_callback(self, ability, successed):
        self.click_history.append([ability,successed])
        self.update()

    def save_button_callback(self):
        self.default_target = self.target.copy()
        self.update()

    def cancel_one(self):
        if len(self.click_history)>0:
            self.click_history.pop()
            self.update()

    def flip_button_callback(self):
        col_0 = [
            self.label_prob,
            self.label_a1,
            self.label_a2,
            self.label_b,
        ]
        col_1 = [
            self.label_chance,
        ]
        col_2 = [
            self.combo_chance,
        ]
        col_3 = [
            self.button_cancel,
            self.button_a1_s,
            self.button_a2_s,
            self.button_b_s,
        ]
        col_4 = [
            self.button_reset,
            self.button_a1_f,
            self.button_a2_f,
            self.button_b_f,
        ]
        col_5 = [
            self.label_pred_title,
            self.label_pred_a1,
            self.label_pred_a2,
            self.label_pred_b,
        ]
        if self.button_right:
            for c0 in col_0:
                c0.grid(column=3)
            for c1 in col_1:
                c1.grid(column=4)
            for c2 in col_2:
                c2.grid(column=5)
            for c3 in col_3:
                c3.grid(column=0)
            for c4 in col_4:
                c4.grid(column=1)
            for c5 in col_5:
                c5.grid(column=2)
            self.button_save.grid(column=4)
            self.label_save.grid(column=5)
            self.button_flip.grid(column=6)
            self.label_target_target.grid(columnspan=3)
        else:
            for c0 in col_0:
                c0.grid(column=0)
            for c1 in col_1:
                c1.grid(column=1)
            for c2 in col_2:
                c2.grid(column=2)
            for c3 in col_3:
                c3.grid(column=3)
            for c4 in col_4:
                c4.grid(column=4)
            for c5 in col_5:
                c5.grid(column=5)
            self.button_save.grid(column=3)
            self.label_save.grid(column=4)
            self.button_flip.grid(column=5)
            self.label_target_target.grid(columnspan=2)
        self.button_right = not self.button_right


    def reset(self, initial=False):
        if initial:
            self.total_chance = 10
            self.a1 = [EMPTY]*10
            self.a2 = [EMPTY]*10
            self.b = [EMPTY]*10
        else:
            self.total_chance = 10- self.combo_chance.current()
        self.click_history = []
        self.target = self.default_target.copy()
        if not initial:
            self.update()

    def update(self):
        self.total_chance = 10-self.combo_chance.current()
        self.prob = 75
        chance_left = [self.total_chance]*3
        target_left = self.target.copy()
        abilities = [[],[],[]]
        buttons=[
            [self.button_a1_s, self.button_a1_f],
            [self.button_a2_s, self.button_a2_f],
            [self.button_b_s, self.button_b_f],
        ]
        target_buttons = [
            self.button_target_a1_list,
            self.button_target_a2_list,
            self.button_target_b_list
        ]
        successes = [0,0,0]
        fails = [0,0,0]
        for ability, successed in self.click_history:
            chance_left[ability] -= 1
            if successed:
                target_left[ability] -= 1
                abilities[ability].append(SUCC)
                successes[ability] += 1
                self.prob_down()
            else:
                abilities[ability].append(FAIL)
                fails[ability] += 1
                self.prob_up()
        for i in range(3):
            for b in buttons[i]:
                if len(abilities[i])==self.total_chance:
                    b.state(['disabled'])
                else:
                    b.state(['!disabled'])
            else:
                for _ in range(chance_left[i]):
                    abilities[i].append(EMPTY)

            # Check if target is possible
            # If impossible, lower target
            if i<2:
                if chance_left[i] < target_left[i]:
                    self.target[i] = chance_left[i] + successes[i]
                    target_left[i] = chance_left[i]
            if i==2:
                # Bad abilities should be the opposite
                if successes[i]>self.target[i]:
                    self.target[i] = successes[i]
                    target_left[i] = 0
            for b in target_buttons[i]:
                b.state(['!disabled'])
            # disable impossible targets
            for s in range(1,successes[i]):
                target_buttons[i][10-s].state(['disabled'])
            for f in range(0,fails[i]+10-self.total_chance):
                target_buttons[i][f].state(['disabled'])

        self.a1, self.a2, self.b = abilities
        self.a1_string_var.set(self.a1_string)
        self.a2_string_var.set(self.a2_string)
        self.b_string_var.set(self.b_string)
        self.prob_string_var.set(self.prob_string)
        self.target_string_var.set(self.target_string(
                                    target_left))
        self.save_string_var.set(self.save_string)

        # Tflite part
        target_left = np.maximum(
            target_left, 0
        )
        # If a2 > a1, flip (for consistency)
        # Although the model is learned regardless of the order
        # Results sometimes do not seem consistent
        # ex) 7/6/2 does not match 6/7/2
        q_input = [
            chance_left[0],
            chance_left[1],
            chance_left[2],
            target_left[0],
            target_left[1],
            target_left[2],
            self.prob_idx
        ]
        if self.table_loaded:
            probs = self.table[
                q_input[0],
                q_input[1],
                q_input[2],
                q_input[3],
                q_input[4],
                q_input[5],
                q_input[6]
            ]
        else:
            probs = get_q(*q_input)
        prob_str_vars = [
            self.pred_a1_string_var,
            self.pred_a2_string_var,
            self.pred_b_string_var
        ]
        recommand = np.argmax(probs)
        for i in range(3):
            if probs[i] == probs[recommand]:
                prob_str_vars[i].set(f'{probs[i]*100:.5f}% (추천)')
            else:
                prob_str_vars[i].set(f'{probs[i]*100:.5f}%')


    def run(self):
        dirname = filedialog.askopenfilename(filetypes=[('넘파이 압축 파일','*.npz')])
        try:
            self.table = np.load(dirname)['table']
        except (IOError, ValueError):
            messagebox.showinfo(message=('테이블 로드에 실패했습니다.\n'
                                '정상 실행은 가능하나\n'
                                '목표 변경시 연산에 따른 딜레이가 생길 수 있습니다.'))
            self.table_loaded = False
            self.explanation_string_var.set(EXPLANATION)
        else:
            self.table_loaded = True

        self.update()
        self.root.mainloop()

if __name__ == '__main__':
    Console().run()