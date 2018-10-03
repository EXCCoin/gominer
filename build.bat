"C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\IDE\devenv.com" .\eqcuda1445\eqcuda1445.sln /Rebuild "Release|x64"

copy .\eqcuda1445\x64\Release\eqcuda1445.dll .\eqcuda1445.dll
copy .\eqcuda1445\x64\Release\eqcuda1445.lib .\eqcuda1445.lib
copy .\eqcuda1445\x64\Release\eqcuda1445.pdb .\eqcuda1445.pdb

dep ensure
go build -ldflags="-extldflags=-static -s -w" -gcflags="-trimpath=%GOPATH%" -asmflags="-trimpath=%GOPATH%"
