CU_APPS=ex1

all: ${CU_APPS}

%: %.cu
	nvcc -O2 -Wno-deprecated-gpu-targets -o $@ $< --expt-relaxed-constexpr
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS}

