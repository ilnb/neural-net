# README

Neural networks and stuff.

# Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release
ln -sf build/compile_commands.json compile_commands.json # for lsp
cmake --build build
```
