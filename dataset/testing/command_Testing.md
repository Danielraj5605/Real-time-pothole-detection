live vlc

tcp/h264://10.91.79.190:8888

rpicam-vid -t 0 \
  --width 640 --height 480 --framerate 30 \
  --inline --listen \
  -o tcp://0.0.0.0:8888



photo

rpicam-still -o test.jpg

ls -lh test.jpg

scp raspberrypi@10.91.79.35:/home/raspberrypi/test.jpg .


video 5 sec

rpicam-vid -t 5000 -o test.h264

ffmpeg -i test.h264 -c copy test.mp4

scp raspberrypi@10.91.79.190:/home/raspberrypi/test.mp4 .
