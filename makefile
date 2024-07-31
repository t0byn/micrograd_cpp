all : demo

demo : main.obj
	link main.obj gvc.lib cgraph.lib /LIBPATH:"Graphviz-12.0.0-win64\lib" /DEBUG:FULL /OUT:demo.exe

main.obj : src/main.cc
	cl /std:c++20 /utf-8 /EHsc /Zi /DGVDLL /I "Graphviz-12.0.0-win64\include" /c src\main.cc /Fo"main.obj"

clean : 
	del *.svg, main.obj, vc140.pdb, demo.pdb, demo.ilk, demo.exe