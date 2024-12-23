OS=$(shell uname -s)
DEBUG=0

INCLUDES=$(wildcard include/*.hpp)
SRC=$(wildcard src/*.cpp)

OBJDIR=./obj/
OBJ=$(addprefix $(OBJDIR), $(notdir $(SRC:.cpp=.o)))
$(shell mkdir -p $(OBJDIR))

DEPS = $(wildcard include/*.hpp)

TARGET=main.out

ifeq ($(OS),Darwin)
	CXX=clang++
	LDFLAGS=-L/opt/homebrew/lib
	COMMON+=-std=c++14 -I/opt/homebrew/include -I/opt/homebrew/include/opencv4
else
	CXX=g++
	LDFLAGS=-L/usr/local/lib
	COMMON+=-I/usr/local/include
endif
LDFLAGS+=-lonnxruntime -lpthread -fopenmp
LDFLAGS+=-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn

ifeq ($(DEBUG),1)
	COMMON+=-g -DDEBUG -DDEBUG_THREAD
else
	COMMON+=-O3
endif


all: $(OBJ)
	$(CXX) $(COMMON) $(OBJ) -o $(TARGET) $(LDFLAGS)

$(OBJDIR)%.o: src/%.cpp $(INCLUDES)
	$(CXX) $(COMMON) -c $< -o $@ -Iinclude

clean:
	rm -f $(OBJDIR)*.o $(TARGET)
