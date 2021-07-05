---
title:  "웹 개발 공부"
#iexcerpt: ""
categories:
  - web_publishing
#tags:
#  - Blog
last_modified_at: 2021-04-29
---



velog.io의 0307kwon님의 글



## 기본 개념

#### 웹 브라우저( 클라이언트)

- 필요한 파일들(html,js,css ...)을 받아 해석하고 사용자에게 보여주는 브라우저



#### 웹 서버

- 클라이언트의 요청(url)에 따라 적절히 응답해주는 프로그램
- 프론트 서버
    - 정적 or 동적인 페이지를 응답하기 위한 서버
- 백엔드 서버
    - 사용자의 요청을 받았을 때 DB에서 적절한 데이터를 가져와 응답하기 위한 서버



#### DB

- 사용자의 목록, 정보 등 중요한 데이터들이 저장된 저장소





### 웹페이지가 동작하는 원리



#### MPA(Multiple Page Application)

- 모든 페이지가 각각의 html로 이루어짐
- 즉, 하나의 페이지에서 다른 페이지로 이동할 때, 반드시 프론트 서버에 요청을 보내고 원하는 페이지의 응답을 받아야 함
- 단점
    - 페이지가 바뀔 때마다 매번 완전한 페이지를 응답 받음
        - 필요한 부분만 응답으로 받는 방식에 비해 비효율적
    - 페이지가 바뀔 때마다 브라우저가 깜빡임
        - 사용자 경험상 좋지 않음



#### SPA(Single Page Application)

- 하나의 페이지(html)에서 모두 처리하는 방식
- 사용자가 초기 url에 접근할 때, 웹에서 이동가능한 모든 페이지에 대한 파일을 클라이언트로 받아 옴 (초기 페이지 로딩이 오래 걸릴 수 있음)
- 페이지 전환 시 프론트 서버에 요청을 보내는 것이 아닌 웹 클라이언트 자체적으로 js에 의해 전환 됨 (js가 라우팅을 담당)
- 페이지 내에 동적으로 변해야 하는 부분이 있는 경우, 해당 정보만 백엔드 서버에 요청하여 응답을 받고 웹에 갱신함