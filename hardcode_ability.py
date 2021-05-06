import tflite_runtime.interpreter as tflite
import numpy as np
import json
from tqdm import trange

model_path = 'tflite_models/stone_1_7.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
input_idx = input_details[0]['index']
output_details = interpreter.get_output_details()
output_idx = output_details[0]['index']

middle = np.array([5,5,5,5,5,5,0.5])
rng = np.array([10,10,10,10,10,10,0.5])

data = np.zeros((11,11,11,11,11,11,6,3),dtype=np.int)
for a1 in trange(1,leave=False):
    for a2 in trange(1,leave=False):
        for a3 in trange(1,leave=False):
            for b1 in trange(1,leave=False):
                for b2 in trange(11,leave=False):
                    for b3 in trange(11,leave=False):
                        for p in trange(6,leave=False):
                            p_p = 0.25+p*0.1
                            tf_input = np.array([[a1,a2,a3,b1,b2,b3,p_p]])
                            tf_input = 2*(tf_input-middle)/rng
                            interpreter.set_tensor(input_idx, tf_input.astype(np.float32))
                            interpreter.invoke()
                            logits = interpreter.get_tensor(output_idx)[0].astype(np.float64)
                            probs = np.exp(logits)/np.sum(np.exp(logits))
                            data[a1,a2,a3,b1,b2,b3,p] = (probs*100).astype(np.int)

data_list = data.tolist()
with open('ability_table.py','w') as f:
    f.write('model = \'')
    json.dump(data_list, f)
    f.write('\'')