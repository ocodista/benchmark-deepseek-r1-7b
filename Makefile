.PHONY: all build clean benchmark

all: build

build:
	go build -o bin/benchmark ./cmd/benchmark
	go build -o bin/monitor ./cmd/monitor

clean:
	rm -rf bin/ 
