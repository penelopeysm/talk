python:
	rm a.out; clang++ -std=c++20 a.cpp -DPYTHON -O3; hyperfine --warmup=2 ./a.out

julia:
	rm a.out; clang++ -std=c++20 a.cpp -DJULIA -O3; hyperfine --warmup=2 ./a.out
