CXX=g++
LDFLAGS=-lonnxruntime `ls /usr/local/lib/*opencv*` -lpthread
OUTPUT=imnet_classifier.out

all:
	$(CXX) imnet_classifier.cpp $(LDFLAGS) -o $(OUTPUT)

clean:
	rm -f imnet_classifier.out