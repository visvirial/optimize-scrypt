CFLAGS  = -ggdb -W -Wall -O2 -std=gnu99
LDFLAGS = -L/usr/local/cuda/targets/x86_64-linux/lib
LDLIBS  = -lpthread -lscrypt -lOpenCL

TARGET = optimize-scrypt
OBJS = $(TARGET).o

.PHONY: all clean

all: $(TARGET)

clean:
	$(RM) $(TARGET) *.o

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

sync:
	rsync -av . nakp:~/src/optimize-scrypt

