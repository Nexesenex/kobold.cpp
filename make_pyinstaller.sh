#!/bin/bash
NUMCPUS=`grep -c '^processor' /proc/cpuinfo` # Get max number of CPU threads
NUMCPUS=$(echo "($NUMCPUS*0.75+0.5)/1" | bc) # Set CPU threads to 3/4th avail. threads, rounding to nearest whole number
printf "\033[33;1mMake sure you've installed OpenCL and OpenBLAS by using \"sudo apt install libclblast-dev libopenblas-dev\"\n\n\n\n\n\n\033[0m\n"
sleep 4
# install dependencies
pip install pyinstaller customtkinter && make clean && \
# Ensure all backends are built then build executable file
make LLAMA_HIPBLAS=1 LLAMA_CLBLAST=1 LLAMA_OPENBLAS=1 -j$NUMCPUS && \
pyinstaller --noconfirm --onefile --clean --console --collect-all customtkinter --collect-all psutil --collect-all libclblast-dev --collect-all libopenblas-dev --collect-all clinfo --icon ".\niko.ico" \
--add-data "./klite.embd:." \
--add-data "./kcpp_docs.embd:." \
--add-data "./kcpp_sdui.embd:." \
--add-data "./taesd.embd:." \
--add-data "./taesd_xl.embd:." \
--add-data "./koboldcpp_default.so:." \
--add-data "./koboldcpp_openblas.so:." \
--add-data "./koboldcpp_failsafe.so:." \
--add-data "./koboldcpp_noavx2.so:." \
--add-data "./koboldcpp_clblast.so:." \
--add-data "./koboldcpp_clblast_noavx2.so:." \
--add-data "./koboldcpp_hipblas.so:." \
--add-data "/opt/rocm/lib/libhipblas.so:." \
--add-data "/opt/rocm/lib/librocblas.so:." \
--add-data "./koboldcpp_vulkan_noavx2.so:." \
--add-data "./koboldcpp_vulkan.so:." \
--add-data "./rwkv_vocab.embd:." \
--add-data "./rwkv_world_vocab.embd:." \
--add-data "/opt/rocm/lib/rocblas:." \
"./koboldcpp.py" -n "koboldcpp_rocm"
