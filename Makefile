OS=$(shell uname -s)
INCLUDES=$(wildcard include/*.hpp)

SRC=$(wildcard src/*.cpp)

OBJDIR=./obj/
OBJ=$(addprefix $(OBJDIR), $(notdir $(SRC:.cpp=.o)))
$(shell mkdir -p $(OBJDIR))

DEPS = $(wildcard include/*.hpp)

TARGET=imnet_classifier.out
LDFLAGS=-lonnxruntime -lpthread

ifeq ($(OS),Darwin)
	CXX=clang++
	LDFLAGS+=-L/opt/homebrew/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn
	COMMON+=-std=c++14 -I/opt/homebrew/include -I/opt/homebrew/include/opencv4
else
	CXX=g++
	LDFLAGS+=`ls /usr/local/lib/*opencv*`
	COMMON+=-I/usr/local/include
endif


all: $(OBJ)
	$(CXX) $(COMMON) $(LDFLAGS) $(OBJ) -o $(TARGET)

$(OBJDIR)%.o: src/%.cpp $(INCLUDES)
	$(CXX) $(COMMON) -c $< -o $@ -Iinclude

clean:
	rm -f $(OBJDIR)*.o $(TARGET)
