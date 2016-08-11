CC = /usr/bin/g++-4.8 -std=c++11 -Wall -O3  -DDEBUG_VERSION
INCL = -I /opt/local/include \
       -I./include_lib \
       -I/opt/local/include/opencv \
       -I/usr/local/include/opencv2/core \


LIB = -L/usr/local/lib/ -lopencv_imgproc -lopencv_highgui -lopencv_core

all:
	$(CC) $(INCL)  draw_shape.C -o draw_shape.x  $(LIB) 
	
