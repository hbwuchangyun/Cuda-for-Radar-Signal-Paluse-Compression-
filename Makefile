#根据运行目标平台选择x86或arm64
TARGET_ARCH = arm64
GPU_ARCH = sm_87
# ARM64 交叉编译器
CROSS_COMPILE := aarch64-none-linux-gnu-g++
#CXX = aarch64-none-linux-gnu-g++
CXX = aarch64-none-linux-gnu-g++
NVCC = nvcc
TARGET = PaluseCompression
#指定编译文件存放路径
BUILD_DIR = build
# 不同目录中的源文件,添加所需要的源代码文件
#CPP_SRCS = src/mat_reader.cpp
CU_SRCS = main.cu src/PaluseCompression.cu
#CPP_SRCS = main.cpp

OBJECTS = $(addprefix $(BUILD_DIR)/,$(CU_SRCS:.cu=.o))
#OBJECTS += $(addprefix $(BUILD_DIR)/,$(CPP_SRCS:.cpp=.o))

#指定单个源文件编译为可执行文件
#OBJECTS = main.o

MATIO_DIR = 3rd/matio/$(TARGET_ARCH)
MATIO_LIB = $(MATIO_DIR)/lib/libmatio.a

INCLUDES += -I $(MATIO_DIR)/include
NVCCFLAGS += -std=c++17 -O2 -Xcompiler=-Wall,-Wextra -MMD -MP
CPPFLAGS += -std=c++17 -O2 -Wall -Wextra -MMD -MP
LDLIBS += -lm -lcufft

INCLUDES += -I src 

#构建AGX上的目标程序时打开
INCLUDES += -I "$(HOME)/jetson-cross/target-cuda/include"
LDLIBS  = -L "$(HOME)/jetson-cross/target-cuda/lib64" -lcudart -lm -lcufft
CROSSFLAG = -ccbin $(CROSS_COMPILE) -arch=$(GPU_ARCH) --cudart=none
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) $(MATIO_LIB)
	$(NVCC) $(CROSSFLAG) -o $@ $(OBJECTS) $(MATIO_LIB) $(LDLIBS)
# 编译 C++ 文件
#$(BUILD_DIR)/%.o: %.cpp
#	mkdir -p $(dir $@)
#	$(CXX) $(INCLUDES) $(CPPFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

#编译.cu文件
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(CROSSFLAG) $(INCLUDES) $(NVCCFLAGS) -c $< -o $@

clean:
	$(RM) $(TARGET) $(OBJECTS) $(OBJECTS:.o=.d)

-include $(OBJECTS:.o=.d)