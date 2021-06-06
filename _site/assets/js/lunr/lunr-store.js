var store = [{
        "title": "알고리즘 카테고리 개설",
        "excerpt":" ","categories": ["algorithm"],
        "tags": [],
        "url": "/algorithm/algorithm-start/",
        "teaser": null
      },{
        "title": "임베디드 시스템 카테고리 개설!",
        "excerpt":" ","categories": ["embedded"],
        "tags": [],
        "url": "/embedded/embedded-start/",
        "teaser": null
      },{
        "title": "웹 개발 공부",
        "excerpt":"velog.io의 0307kwon님의 글 기본 개념 웹 브라우저( 클라이언트) 필요한 파일들(html,js,css …)을 받아 해석하고 사용자에게 보여주는 브라우저 웹 서버 클라이언트의 요청(url)에 따라 적절히 응답해주는 프로그램 프론트 서버 정적 or 동적인 페이지를 응답하기 위한 서버 백엔드 서버 사용자의 요청을 받았을 때 DB에서 적절한 데이터를 가져와 응답하기 위한 서버 DB 사용자의 목록, 정보...","categories": ["web_publishing"],
        "tags": [],
        "url": "/web_publishing/web_publishing/",
        "teaser": null
      },{
        "title": "pandas 사용법",
        "excerpt":"Pandas 기본 사용법 # 판다스 라이브러리 import import pandas as pd # 파일로부터 데이터 읽어오기 file_path = 'lemonade.csv' lemonade = pd.read_csv(file_path) lemonade # 데이터 모양 확인하기 print(lemonade.shape) (6, 2) # 칼럼이름 출력 print(lemonade.columns) Index([‘온도’, ‘판매량’], dtype=’object’) # 칼럼 독립변수, 종속변수로 분리 indep = lemonade[['온도']] # 독립 변수 dep = lemonade[['판매량']]...","categories": ["machine_learning"],
        "tags": [],
        "url": "/machine_learning/ml_pandas/",
        "teaser": null
      },{
        "title": "개발환경 구축",
        "excerpt":"개발환경 구축 필요한 이유 독립적인 작업환경에서 작업하기 위함이다. 프로젝트 진행에 있어 여러 라이브러리, 패키지 다운로드에 있어 각 라이브러리들끼리 충돌 또는 특정 버전과 호환문제가 발생할 수 있다. 이를 방지하기 위해 프로젝트 단위로 가상환경을 구성하여 필요한 라이브러리를 설치해서 작업한다. 이를 통해 다른 컴퓨터 혹은 다른 환경에서 동일한 프로그램을 실행시킬 때, 작업환경을 고정시켰으므로...","categories": ["machine_learning"],
        "tags": [],
        "url": "/machine_learning/ml_env/",
        "teaser": null
      },{
        "title": "EX1_regression",
        "excerpt":"EX1 라이브러리 사용 import pandas as pd import tensorflow as tf 1. 데이터 준비 # 1. 과거의 데이터 준비 레몬에이드 = pd.read_csv('lemonade.csv') 레몬에이드 # 독립변수/종속변수 분리 독립 = 레몬에이드[['온도']] 종속 = 레몬에이드[['판매량']] print(독립.shape, 종속.shape) (6,1) (6,1) 2. 모델의 구조 만들기 # 2. 모델의 구조 만들기 X = tf.keras.layers.Input(shape=[1]) # 독립...","categories": ["machine_learning"],
        "tags": [],
        "url": "/machine_learning/ml_ex1/",
        "teaser": null
      },{
        "title": "EX2_classification",
        "excerpt":"EX2 라이브러리 사용 import pandas as pd import tensorflow as tf 데이터 준비 파일경로 = 'iris.csv' 아이리스 = pd.read_csv(파일경로) 아이리스.head() one-hot encoding 인코딩 = pd.get_dummies(아이리스) 인코딩.head() 독립변수, 종속변수 print(인코딩.columns) 독립 = 인코딩[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']] 종속 = 인코딩[['품종_setosa', '품종_versicolor', '품종_virginica']] print(독립.shape, 종속.shape) Index([‘꽃잎길이’, ‘꽃잎폭’, ‘꽃받침길이’, ‘꽃받침폭’, ‘품종_setosa’, ‘품종_versicolor’, ‘품종_virginica’], dtype=’object’)...","categories": ["machine_learning"],
        "tags": [],
        "url": "/machine_learning/ml_ex2/",
        "teaser": null
      }]
