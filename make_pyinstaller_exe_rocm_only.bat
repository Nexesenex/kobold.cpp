cd /d "%~dp0"
copy "C:\Program Files\AMD\ROCm\5.7\bin\hipblas.dll" .\ /Y
copy "C:\Program Files\AMD\ROCm\5.7\bin\rocblas.dll" .\ /Y
xcopy /E /I "C:\Program Files\AMD\ROCm\5.7\bin\rocblas" .\rocblas\ /Y
curl -LO https://github.com/YellowRoseCx/koboldcpp-rocm/releases/download/v1.43.2-ROCm/gfx103132rocblasfiles.zip
tar -xf gfx103132rocblasfiles.zip -C .\ --strip-components=1
python -m pip install cmake ninja pyinstaller==6.4.0 psutil customtkinter
 
PyInstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all psutil --icon "./niko.ico" --add-data "./winclinfo.exe;." --add-data "./klite.embd;." --add-data "./kcpp_docs.embd;." --add-data="./kcpp_sdui.embd;." --add-data="./taesd.embd;." --add-data="./taesd_xl.embd;." --add-data "./rwkv_vocab.embd;." --add-data "./rwkv_world_vocab.embd;." --add-data "./koboldcpp_hipblas.dll;." --add-data "C:/Program Files/AMD/ROCm/5.7/bin/hipblas.dll;." --add-data "C:/Program Files/AMD/ROCm/5.7/bin/rocblas.dll;." --add-data "C:/Program Files/AMD/ROCm/5.7/bin/rocblas;." --add-data "C:/Windows/System32/msvcp140.dll;." --add-data "C:/Windows/System32/vcruntime140_1.dll;." "./koboldcpp.py" -n "koboldcpp_rocm.exe"
