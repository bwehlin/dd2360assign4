
ex3.out: exercise_3.cpp
	pgc++ -O3 -std=c++11 -acc -ta=tesla:cc3x,nvidia,lineinfo -Minfo=accel -o ex3.out exercise_3.cpp

.PHONY: all
all: ex3.out
