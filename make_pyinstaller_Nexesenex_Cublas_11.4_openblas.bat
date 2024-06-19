cd /d "%~dp0"
copy "P:\NVIDIAGPUCT\CUDA\v11.4\bin\cudart64_110.dll" .\ /Y
copy "P:\NVIDIAGPUCT\CUDA\v11.4\bin\cublasLt64_11.dll" .\ /Y
copy "P:\NVIDIAGPUCT\CUDA\v11.4\bin\cublas64_11.dll" .\ /Y
PyInstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all psutil --icon ".\nikogreen.ico" --add-data "./winclinfo.exe;." --add-data "./klite.embd;." --add-data "./kcpp_docs.embd;." --add-data="./kcpp_sdui.embd;." --add-data="./taesd.embd;." --add-data="./taesd_xl.embd;." --add-data "./koboldcpp_openblas.dll;." --add-data "./koboldcpp_cublas.dll;." --add-data "./cudart64_110.dll;." --add-data "./cublasLt64_11.dll;." --add-data "./cublas64_11.dll;." --add-data "./msvcp140.dll;." --add-data "./msvcp140_codecvt_ids.dll;." --add-data "./vcruntime140.dll;." --add-data "./vcruntime140_1.dll;." --add-data "./rwkv_vocab.embd;." --add-data "./rwkv_world_vocab.embd;." "./koboldcpp.py" -n "koboldcpp_cuda.exe"
pause