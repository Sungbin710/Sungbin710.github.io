---
title:  "TF Lite 개발"
#iexcerpt: ""
categories:
  - machine_learning
#tags:
#  - Blog
last_modified_at: 2021-06-29
---

## TF Lite 개발 실습



### 01 모델 생성하기

- Tensor Flow 모델의 생성,학습 및 추론을 수행하고 모델의 저장 과정을 배워보자.

#### 라이브러리 추가

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
```



#### 1. 임의의 데이터 생성

```python
#1.make data
test_size = 100

_input  = np.arange(100, 100+test_dimsize)
_output = _input * 100 + 5
```

- numpy를 이용해 100부터 차례로 test_size 만큼 array를 생성하여 _input으로 할당한다.
- _output은 _input에 100을 곱하고 5를 더하여 y=x*100+5의 함수형태로 표현된다.



#### 2. 모델 선언 및 요약

```python
#2.construct layer
model = keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1]),
    tf.keras.layers.Dense(1)
    ])
    
model.compile(optimizer=keras.optimizers.Adam(0.01),
        loss='mse')

model.summary()
```

- keras의 Sequential class를 사용해서 Dense Layer를 2개 쌓았다. 
- 첫 번째 레이어는 unit 개수는 10개, 두 번째 레이어의 unit 개수는 1개로 구성하였다.

![summary](\assets\images\machine_learning\TF_LITE\summary.png)

- model.summary()를 통해 생성한 모델의 결과를 볼 수 있다.





#### 3. 모델 학습 및 weight 확인

```python
#3.train model
model.fit(_input, _output, epochs=2000, batch_size=10, verbose=1)

weight = model.get_weights()
print("weight: ", weight)
```

![training](\assets\images\machine_learning\TF_LITE\training.png)

- 학습을 수행한다.
- verbose는 default로 1로 지정되어 있고, 1이면 학습이 진행되는 것을 시각적으로 보이게 표현, 0이면 표현하지 않음

![weight](\assets\images\machine_learning\TF_LITE\weight.png)

- weight 값을 직접 확인할 수 있다.



#### 4. 추론

```python
#4.Inference
print("prediction test")
test_input = [1,2,3,4,5]
print("input : ", test_input)
test_output = model.predict(test_input)
print("output: ", test_output.tolist())
```

![prediction](\assets\images\machine_learning\TF_LITE\prediction.png)

- 학습을 통해 학습하고자 한 함수 _output = _input * 100 + 5에 근접한 결과를 확인한다.
- 학습이 완료되면, 학습된 weight 값들을 잃어버러지 않게 모델을 저장해준다.



#### 5. 모델 저장

```python
#5. save model as tensorflow saved_model
model.save('saved_model/my_model')
```

![save](\assets\images\machine_learning\TF_LITE\save.png)

- 지정한 경로에 pb 파일과 2개의 폴더로 모델 정보를 저장한 것을 확인할 수 있다.







### 02 모델 로드하여 사용하기

- 사전에 학습된 모델을 로드하고 추론을 수행해보자.

#### 라이브러리 추가

```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
```



#### 1. TF 모델 로드하기

```python
#1. load TF model
saved_path = 'saved_model/my_model'
print("loaded TF model %s" % saved_path)
loaded = tf.keras.models.load_model(saved_path)
```

- 모델을 저장해둔 경로를 지정하여 모델 로드를 진행한다.



#### 2. 추론

```python
#2. predict
print("prediction test")
test_input = [1,2,3,4,5]
print("input : ", test_input)
test_output = loaded.predict(test_input)
print("output: ", test_output.tolist())
```

- 01 simple-regression에서 수행했을 때와 동일한 추론 결과값을 보여준다.
- 동일한 weight 값들을 사용했기 때문에 당연한 결과이다.



#### 3. weight 확인

```python
#3. print model weights
weight = loaded.get_weights()
print("weight len : %d" % len(weight))
for i in range(len(weight)):
    print("weight[%d]" % i)
    print(weight[i])
    print("")
```

![weight2](\assets\images\machine_learning\TF_LITE\weight2.png)

- Dense Layer의 weight와 bias가 순서대로 출력된다.



#### 4. 로드한 모델을 그래프로 저장

```python
#4. save a model as png
tf.keras.utils.plot_model(
    loaded, to_file='saved_model/my_model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
) 
```

- 모델을 그래프로 그려주는 함수를 사용하여, 간략히 모델 그래프를 그려볼 수 있다.
- 아래와 같은 png 파일을 생성하여 준다.

![model_image](\assets\images\machine_learning\TF_LITE\model_image.png)





### 03 모델 변환하기

- TensorFlow로 생성한 모델 파일을 TF Lite 모델 파일로 변환하는 과정을 배워보자.
- 이 변환과정이 필요한 이유는 TensorFlow는 Protocol Buffers를 사용해서 pb파일로 모델을 저장하는데 반해, TF Lite는 FlatBuffers를 사용해서 tflite 파일로 저장하기 때문이다.
- FlatBUffers가 Protocol Buffers 보다 작고 가볍기 때문에 TF Lite에 적합하다.
- TF Lite는 IoT나 Embedded 환경을 위해서 설계되었기 때문에 빠른 성능을 위해서 작고 가벼운 포맷을 필요로 한다.
- 여기서는 두 가지 방법을 통해 모델을 변환을 수행해 본다.



#### 1.1. command line tool(tflite_convert)

```bash
tflite_convert \
  --saved_model_dir=./saved_model/my_model \
  --output_file=./saved_model/my_model.cmdline.tflite
```

- convert-tflite.sh 파일에 위와 같이 tflite_convert 기본 명령 옵션을 적어둔다.
- sh파일을 실행하여 output_file에 지정한 경로에 tflite 파일이 생성된 것을 확인한다.
- window에서는 sh convert-tflite.sh 명령어를 통해 sh파일을 실행할 수 있다.



#### 1.2. Python API Converter(TFLiteConverter)

##### 1.2.1 라이브러리 추가

```python
import tensorflow as tf
```



##### 1.2.2 모델 변환

```python
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
tflite_model = converter.convert()
open("saved_model/my_model.tflite", "wb").write(tflite_model)
```

- Python API는 tensorflow.lite.TFLiteConverter 클래스를 사용한다.
- TensorFlow 모델을 로드하고 convert를 실행하면 tflite 포맷으로 저장이 된다.



##### 1.2.3 최적화

```python
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model/my_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
open("saved_model/my_model.quant.tflite", "wb").write(tflite_quant_model)
```

- Python API는 양자화(quantization) 옵션을 제공한다. 양자화는 float32로 학습된 weight 값들을 단순화 시키는 작업을 의미한다.
- float32로 학습된 결과를 이보다 작은 data 타입인 float16, int8등의 형태로 변환하는 것이다.
- 해당 변환의 대부분은 세밀한 데이터값을 버리는 작업이므로 전체적으로 보면 모델의 정확도를 잃어버리는 작업이 된다.
- 정확도를 잃으면서 작은 모델 사이즈를 얻게되고, 작은 모델 사이즈는 추론시간을 줄여준다.
- 즉, 양자화는 모델 사이즈를 줄여서 정확도는 떨어지느 대신에 빠른 추론시간을 얻게 되는 작업이다.
- 해당 예시에서는 float16으로 양자화를 수행하였다.





### 04 모델 배포하기

- 변환 과정을 완료한 TF Lite 모델을 이용하여 추론을 실행해보자.
- TF Lite 모델은 IoT, Embedded 기기 등에서 실행되는 것이 목적이고, 이를 위해 여러가지 플랫폼과 언어를 사용할 배포를 지원한다. 주로 안드로이드와 iOS 플랫폼에서 TF Lite 추론을 지원하고, 언어는 C++, Java, Python 등을 사용할 수 있다.



#### 라이브러리 추가

```python
# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
```



#### 1. TF Lite 모델 로드 및 텐서 초기화

```python
# Load TFLite model and allocate tensors.
#interpreter = tf.lite.Interpreter(model_path="./saved_model/my_model.quant.tflite")
interpreter = tf.lite.Interpreter(model_path="./saved_model/my_model.tflite")
interpreter.allocate_tensors()
```





#### 2. 입력 데이터 생성

```python
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("input details: ",input_details)

# Test model on random input data.
input_shape = input_details[0]['shape']
#input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
input_data = np.array(np.random.randint(0,1000, size=input_shape), dtype=np.float32)
```

![input_details](\assets\images\machine_learning\TF_LITE\input_details.png)

- tensorflow.lite.Interpreter 클래스의 get_input_details() 와 get_output_details()를 통해 세부 정보를 받아온다. 
- 랜덤하게 input_shape 만큼의 입력 데이터를 생성하여 input_data에 저장한다.



#### 3. 추론 실행 및 결과 확인

```python
print("input : %s" % input_data)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print("output : %s" % output_data)
```

- input_data 값을 확인한다.
- input_data를 tflite model의 input으로 할당하고, invoke()를 통해 추론을 실행한다.

![inference](\assets\images\machine_learning\TF_LITE\inference.png)

- 추론을 실행한 후에, get_tensor() 함수를 사용하여 output 텐서에 저장되어 있는 값을 불러와서 출력한다.