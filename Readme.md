# micrograd_cpp
This is my attempt to write a tiny autograd engine in C++ following Andrej Karpathy's video: [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0).

The code is a little messy, could be improve. There are still some features and examples that need to be added compared to [karpathy's micrograd](https://github.com/karpathy/micrograd).

Also I haven't implemented the visualization for expression graphs.

## build
```
cl /std:c++20 /EHsc /Zi .\src\main.cc /link /DEBUG:FULL /OUT:test.exe
```

## run
```
.\test.exe
```

## debug
```
devenv /DebugExe .\test.exe
```
