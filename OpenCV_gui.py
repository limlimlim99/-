#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


# 이미지 다운로드
# urllib.request 모듈은 웹사이트 데이터 접근
import urllib.request as req
import matplotlib.pyplot as plt
import cv2

url = 'https://image.shutterstock.com/z/stock-photo-happy-cheerful-young-woman-wearing-her-red-hair-\
in-bun-rejoicing-at-positive-news-or-birthday-gift-613759379.jpg'

# download
req.urlretrieve(url, 'lady.png')

# OpenCV로 읽어 들이기
img = cv2.imread('lady.png')

# 이미지 출력
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # OpenCV의 BGR 형태를 matplot형태인 RGB로 변환해준다!
plt.show()


# In[3]:


# 화면 잘나오는지 확안
import cv2
path = "C:\lim\Lenna.png"
image = cv2.imread(path)
window_name = 'image'
cv2.imshow(window_name, image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[4]:


path = "C:\lim\Lenna.png"
image = cv2.imread(path,0)  # 흑백 사진
window_name = 'image'
cv2.imshow(window_name, image)
cv2.waitKey(0) # 아무키를 누를때까지 기다림
cv2.destroyAllWindows()  # closing all open windows


# In[5]:


path = "C:\lim\Lenna.png"
image = cv2.imread(path)

x,y,w,h = cv2.selectROI('image',image, False)
if w and h:
    roi = image[y:y+h, x:x+w]
    cv2.imshow('drag',roi)
    cv2.imwrite('drag.jpg',roi)
    
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()  # 드래그 후 Enter 해야 저장됨


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[3]:


get_ipython().run_line_magic('cd', 'C:/lim/data')


# In[7]:


import cv2

image = cv2.imread('pattern_2-1-8-10-D.jpg')
print(image.shape)
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('image', image) # 왼쪽은 window 이름, 오른쪽은 이미지
cv2.waitKey()  ## 키 누를때까지 기다림
cv2.destroyAllWindows() 


# In[30]:


# 마우스로 좌표 위치를 확인
image = cv2.imread('pattern_2-1-8-10-D.jpg')
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x, y, w, h = cv2.selectROI('image', image, False)
if w and h:
    roi = image[y:y+h, x:x+w]
    print(y, y+h, x, x+w) # 좌표를 프린트 해줌
    cv2.imwrite('drag.jpg', roi)   #roi를 drag.jpg에 저장해줌

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()

# 32 186 73 225
# 233 1002 260 1000
# 23 187 515 766
# 248 302 642 695
# 502 688 507 681
# 62 104 98 149

# y, y+h, x, x+w


# In[32]:


# 마우스로 좌표 위치를 확인
image = cv2.imread('pattern_2-1-8-10-D.jpg')
cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

imagebound = image.copy()
blue = (255 ,0, 0)

#cv2.rectangle(imagebound, (73, 32), (255, 186), blue, 2)
#cv2.rectangle(imagebound, (260, 233), (1000, 1002), blue, 2)
#cv2.rectangle(imagebound, (515, 23), (766, 187), blue, 2)
#cv2.rectangle(imagebound, (642, 248), (695, 302), blue, 2)
#cv2.rectangle(imagebound, (507, 502), (681, 688), blue, 2)
cv2.rectangle(imagebound, (98, 62), (149, 104), blue, 2)

cv2.imshow('image', imagebound)
cv2.waitKey()  
cv2.destroyAllWindows() 


# In[33]:


for i in range(0, 1119, 98):
    for j in range(0,119-62, 62):
        blue[i:98+i, j:62+j] = blue
cv2_imshow(blue)


# In[44]:


x =91; y=60; w=98; h=42;
roi = image[y:y+h, x:x+w]
image[62:104, 98:149] = roi
cv2.rectangle(imagebound, (x,y), (x+w+w, y+h), (255,0,0))
cv2.imshow('image', imagebound)
cv2.waitKey() 
cv2.destroyAllWindows() 


# In[43]:


import matplotlib.pyplot as plt
t =image[62:104,98:149]
plt.imshow(t)


# In[ ]:


for i in range(0, 646-70, 70):
  for j in range(0, 960-100, 100):
    tree[i: 70+i, j: 100+j] = t
cv2_imshow(tree)  


# In[ ]:





# # 동영상

# In[45]:


get_ipython().system('pip install wget')


# In[48]:


import wget
url = 'https://github.com/chulminkw/DLCV/blob/master/data/video/Night_Day_Chase.mp4?raw=true'
wget.download(url)


# In[66]:


# 동영상 불러오기
# cap.read() 영상 정보 읽기, cap.get() 옵션으로 다른 정보만 추출해서 영상 정보 읽기
import cv2

cap = cv2.VideoCapture('Night_Day_Chase.mp4')   # 영상이 저장되어있음
cap
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
    #if cv2.waitKey(10) & 0xFF == ord('q'):   #q를 누르면 꺼짐
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 


# In[59]:


cap = cv2.VideoCapture('Night_Day_Chase.mp4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  ## (200, 400)
count = cap.get((cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print('가로: ', str(width))
print('세로: ', str(height))
print('총 프레임수: ', str(count))
print('FPS: ', str(fps))


# OpenCV를 활용한 영상 처리
# - OpenCV의 VideoCapture 클래스
#   - 동영상을 개별 Frame으로 하나씩 읽어들이는 기능 제공
#   - 생성 인자로 입력 video 파일 위치를 받아 생성
#     - cap=cv2.VideoCapture(video_input_path)
#   - 입력 video 파일의 다양한 속성 가져오기 가능
#     - 영상 Frame 너비
#       cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     - 영상 Frame 높이
#       cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     - 영상 FPS(Frame Per Second)
#       cap.get(cv2.CAP_PROP_FPS)
#   - read()는 마지막 Frame까지 차례로 Frame을 읽음
# - VideoWriter
#   - VideoCapture로 읽어들인 개별 Frame을 동영상 파일로 Write 수행
#   - write할 동영상 파일 위치, Encoding 코덱 유형, write fps 수치, frame 크기를 생성자로 입력 받음
#   - 이들 값에 따른 동영상 write 수행
#   - write 시, 특정 포맷으로 동영상 Encoding 가능
#     - DIVX, XVID, MJPG, X264, WMV1, WMV2
# 

# In[63]:


# VideoCapture 객체와 Videowriter 객체를 생성하고 frame의 갯수, Video frame크기의 정보를 읽어 출력
import cv2

video_input_path = 'Night_Day_Chase.mp4'
video_output_path = 'Night_Day_Chase_out.mp4'

cap = cv2.VideoCapture(video_input_path)

codec = cv2.VideoWriter_fourcc(*'XVID')

vid_size = (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vid_fps = cap.get(cv2.CAP_PROP_FPS)

vid_writer = cv2.VideoWriter(video_output_path, codec, vid_fps, vid_size)

frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('총 Frame 갯수:', frame_cnt, 'FPS:', round(vid_fps), 'Frame 크기 :', vid_size)


# In[68]:


import time

green_color=(0, 255, 0)
red_color=(0, 0, 255)

start = time.time()
index = 0
# 프레임 하나씩 일고 쓰기
while True:
    hasFrame, img_frame = cap.read()
    if not hasFrame:
        print('더 이상 처리할 frame이 없습니다.')
        break
    index += 1
    print('frame :', index, '처리 완료')
    
    cv2.rectangle(img_frame, (300, 100, 800, 400), color=green_color, thickness=2)
    caption = 'frame:{}'.format(index)
    cv2.putText(img_frame, caption, (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, red_color, 1)
    
    vid_writer.write(img_frame)
    
print('write 완료 시간:', round(time.time() - start, 4))
vid_writer.release()
cap.release()


# In[72]:


import cv2
cap = cv2.VideoCapture('Night_Day_Chase.mp4')
cap
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 


# In[ ]:




