import tflite_runtime.interpreter as tflite
import tkinter as tk
from tkinter import ttk
from functools import partial
import numpy as np

EMPTY = "◇"
SUCC = "◆"
FAIL = "✕"
MAX_PROB = 75
MIN_PROB = 25
DEFAULT_TARGET = [7,6,2]
MODEL_PATH = 'tflite_models/stone_1_7.tflite'

class Console():
    def __init__(self):
        self.prepare_tflite()

        self.root = tk.Tk()
        self.root.title('AbilityStone')
        self.mainframe = ttk.Frame(self.root, padding='3 3 12 12')
        self.mainframe.grid(column=0, row=0, sticky=(tk.NSEW))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.resizable(False,False)

        self.default_target = DEFAULT_TARGET
        self.reset(initial=True)

        self.click_history = []
        
        self.target_string_var = \
            tk.StringVar(value=self.target_string(self.target,self.target))
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

        # self.font_prob = 
        self.label_prob = ttk.Label(self.mainframe,
                            textvariable=self.prob_string_var,
                            padding='5 5 5 5',
                            font=(None,16,'bold'))
        self.label_prob.grid(column=0, row=0)
        self.label_chance = ttk.Label(
            self.mainframe,
            text='칸 수: '
        )
        self.label_chance.grid(column=1, row=0)
        self.combo_chance = ttk.Combobox(
            self.mainframe,
            values=[
                10,9,8,7,6
            ],
            state='readonly'
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
            text='로파고의 확신'
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
            column=0, columnspan=2,row=4
        )
        self.label_target_target = ttk.Label(
            self.mainframe,
            textvariable=self.target_string_var,
            font=(None,14,'bold')
        )
        self.label_target_target.grid(column=2, row=4)

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
        for i in range(10):
            self.button_target_a1_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 0, 10-i)
                )
            )
            self.button_target_a1_list[-1].grid(column=0, row=i+1)


        self.button_target_a2_list = []
        for i in range(10):
            self.button_target_a2_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 1, 10-i)
                )
            )
            self.button_target_a2_list[-1].grid(column=1, row=i+1, padx=10)

        self.button_target_b_list = []
        for i in range(10):
            self.button_target_b_list.append(
                ttk.Button(
                    self.frame_target,
                    text=10-i,
                    command=partial(self.target_button_callback, 2, 10-i)
                )
            )
            self.button_target_b_list[-1].grid(column=2, row=i+1)
        
        self.update()

    def prepare_tflite(self):
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.input_idx = self.input_details[0]['index']
        self.output_details = self.interpreter.get_output_details()
        self.output_idx = self.output_details[0]['index']


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

    def target_string(self, target, target_left):
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
            self.target, target_left
        ))
        self.save_string_var.set(self.save_string)

        # Tflite part
        target_left = np.maximum(
            target_left, 0
        )
        tf_input = np.float32(np.concatenate(
            (chance_left, target_left, [self.prob/100])
        ))[np.newaxis,...]
        # Normalize inputs
        high = np.array([10,10,10,10,10,10,1],dtype=np.float32)
        low = np.array([0,0,0,0,0,0,0],dtype=np.float32)
        obs_range = high-low
        obs_middle = (high+low)/2
        tf_input = 2*(tf_input-obs_middle)/obs_range
        self.interpreter.set_tensor(self.input_idx,tf_input)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_idx)[0].astype(np.float64)
        probs = np.exp(logits)/np.sum(np.exp(logits))
        if chance_left[0] == 0:
            probs[1] += probs[0]
            probs[0] = 0

        if chance_left[1] == 0:
            if chance_left[0] > 0:
                probs[0] += probs[1]
            else:
                probs[2] += probs[1]
            probs[1] = 0
        
        if chance_left[2] == 0:
            if chance_left[0] > 0:
                probs[0] += probs[2]
            else:
                probs[1] += probs[2]
            probs[2] = 0

        prob_str_vars = [
            self.pred_a1_string_var,
            self.pred_a2_string_var,
            self.pred_b_string_var
        ]
        for sv, prob in zip(prob_str_vars, probs):
            sv.set(f'{prob*100:.2f}%')



    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    Console().run()