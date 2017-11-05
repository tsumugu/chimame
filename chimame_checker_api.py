# -*- coding:utf-8 -*-
import keras_chimame as chimame
import sys
import os
import shutil
import cv2
import numpy as np
import tornado.web
import tornado.ioloop
from PIL import Image

image_size = 50
categories = ["Chino", "Maya", "Megu", "Other"]
#B,G,R
color_chino = (250, 206, 135)
color_maya = (178, 58, 96)
color_megu = (67, 31, 223)
color_other = (0, 165, 255)
theme_colors = [color_chino,color_maya,color_megu,color_other]
fontType = cv2.FONT_HERSHEY_SIMPLEX
X = []
files = []
face_pos = []
name = []
tcolors = []

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        #画像保存処理
        """
        try:
            if len(sys.argv[1]) == 0:
                filename = "test.jpg"
            else:
                filename = sys.argv[1]
        except IndexError:
            filename = "test.jpg"
        """
        dl_img_url = str(self.get_argument('url'))

        if os.path.isfile(filename):
            #顔切り出し、切り出したファイル名突っ込む
            charas = []
            cascade_path =  "lbpcascade_animeface.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            img_src = cv2.imread(filename)
            # グレースケールに変換
            img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
            faces  =  cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100))
            # 顔があった場合
            i=0
            if len(faces) > 0:
                # 複数の顔があった場合、１つずつ保存
                for face in faces:
                    x = face[0]
                    y = face[1]
                    width = face[2]
                    height = face[3]
                    dst = img_src[y:y+height, x:x+width]
                    face_pos.append(face)
                    #200x200にリサイズ
                    resized = cv2.resize(dst,(200,200))
                    new_image_path = 'faces/anime_' + str(i) + ".jpg"
                    charas.append(new_image_path)
                    cv2.imwrite(new_image_path, resized)
                    i += 1
            if i == 0:
                #print("顔が見つかりませんでした。")
                self.write("顔が見つかりませんでした。")
            else:
                for fname in charas:
                    img = Image.open(fname)
                    img = img.convert("RGB")
                    img = img.resize((image_size,image_size))
                    in_data = np.asarray(img)
                    X.append(in_data)
                    files.append(fname)
                X = np.array(X)
        
                model = chimame.build_model(X.shape[1:])
                model.load_weights("chimame-model.hdf5")
        
                pre = model.predict(X)
                for i, p in enumerate(pre):
                    y = p.argmax()
                    #print("ファイル名:",files[i])
                    #print("キャラ名:",categories[y])
                    name.append(categories[y])
                    tcolors.append(theme_colors[y])
        
                #顔に四角&名前
                for i, ps in enumerate(face_pos):
                    print("キャラ名:",name[i])
                    print("顔座標:",ps)
                    #色設定
                    color = tcolors[i]
                    # 囲う四角の左上の座標
                    coordinates = tuple(ps[0:2])
                    # (囲う四角の横の長さ, 囲う四角の縦の長さ)
                    length = tuple(ps[0:2] + ps[2:4])
                    #書き込み
                    cv2.rectangle(img_src, coordinates, length, color, thickness=3)
                    cv2.putText(img_src, name[i], (ps[0], ps[1]), fontType, 3, color, 3, 3)
                cv2.imwrite("res_"+filename, img_src)
        
            #フォルダ作り直し
            shutil.rmtree("faces")
            os.mkdir("faces")
        else:
            #print("エラー：ファイル("+filename+")が存在しません")
            self.write("エラー：ファイル("+filename+")が存在しません")

application = tornado.web.Application([
    (r"/", MainHandler),
])

if __name__ == "__main__":
    application.listen(80)
    tornado.ioloop.IOLoop.instance().start()