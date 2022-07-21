all: aos-ocl.exe soa-ocl.exe aos-omp.exe soa-omp.exe aos.exe soa.exe generator.exe

soa: soa-ocl.exe soa-omp.exe soa.exe

aos: aos-ocl.exe # aos-omp.exe aos.exe

CC=g++

CFLAGS= -std=c++17 -g -march=native -lOpenCL -fopenmp

%.o: src/%.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

%.exe: src/%.cpp
	$(CC) $(CFLAGS) $< -o $@

# seq
aos.exe: src/main.cpp kmeansAOS.o
	$(CC) $(CFLAGS) $< -o $@ kmeansAOSp.o

soa.exe: src/main.cpp kmeansSOA.o
	$(CC) $(CFLAGS) $< -o $@ kmeansSOA.o

# omp
aos-omp.exe: src/main.cpp kmeansAOSp.o
	$(CC) $(CFLAGS) $< -o $@ kmeansAOSp.o

soa-omp.exe: src/main.cpp kmeansSOAp.o
	$(CC) $(CFLAGS) $< -o $@ kmeansSOAp.o

# ocl
aos-ocl.exe: src/main.cpp kmeansAOSCL.o  OpenClinit.o
	$(CC) $(CFLAGS) $< -o $@ kmeansAOSCL.o OpenClinit.o

soa-ocl.exe: src/main.cpp kmeansSOACL.o  OpenClinit.o
	$(CC) $(CFLAGS) $< -o $@ kmeansSOACL.o OpenClinit.o

clean:
	rm *.exe *.o

testrun= SHELL:=/bin/bash
testrun: aos-ocl.exe
	for i in {1..10}; do \
		./aos-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv; \
	done

perfrun: aos-ocl.exe
	perf stat --repeat 5 ./aos-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv

testaos: aos-ocl.exe
	./aos-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv

testsoa: soa-ocl.exe
	./soa-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv

test-omp:
	./soa-omp.exe 5 0.000005 data/02_Skin_NonSkin.csv

generate: generator.exe 
	./generator.exe 10 2 500 > data/10-2-500.csv

runocl: SHELL:=/bin/bash

runocl: soa-ocl.exe  aos-ocl.exe 
	for i in {1..10}; do \
		./aos-ocl.exe 5 0.000005 data/01_iris.csv >> AOS-01ocl.csv; \
		./aos-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv >> AOS-02ocl.csv; \
		./soa-ocl.exe 5 0.000005 data/01_iris.csv >> SOA-01ocl.csv; \
		./soa-ocl.exe 5 0.000005 data/02_Skin_NonSkin.csv >> SOA-02ocl.csv; \
	done
