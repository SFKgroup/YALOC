from re import T
import cv2

worng_detect = {}
wrong_answer = {}

grand = open('./error.txt','r',encoding='utf-8')
for i in grand.readlines():
    lst = i.replace('\n','').split(' ')
    img = cv2.imread(lst[1])
    cv2.putText(img, lst[3], (20,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    cv2.putText(img, lst[5], (20,60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    try:worng_detect[lst[5]]+=1
    except:worng_detect[lst[5]]=1
    try:wrong_answer[lst[3]]+=1
    except:wrong_answer[lst[3]]=1
    cv2.imshow('show',img)
    cv2.waitKey(0)

print('wrong_answer',wrong_answer)

print('worng_detect',worng_detect)

