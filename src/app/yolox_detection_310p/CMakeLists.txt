# 定义项目名称
project(yolox_detection_310p)

# GryFlux 要求至少使用 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================
# 1. 查找依赖包 (OpenCV & Ascend ACL)
# ==========================================
find_package(OpenCV REQUIRED)

# 配置 Ascend ACL 路径
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend Toolkit Path")

# ==========================================
# 2. 收集源文件并生成可执行文件
# ==========================================
# 收集你写的所有的 .cpp 文件
file(GLOB_RECURSE APP_SOURCES
    "*.cpp" 
)

# 【关键修复】必须把 ${SRC_DIR} (框架源码) 也加进来到一起编译
add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    ${SRC_DIR}
)

# ==========================================
# 3. 包含头文件目录
# ==========================================
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}         # 包含当前目录
    ${OpenCV_INCLUDE_DIRS}              # OpenCV 头文件
    ${ASCEND_PATH}/include              # ACL 头文件
)

# ==========================================
# 4. 链接库
# ==========================================
# 添加 ACL 动态库的搜索路径
target_link_directories(${PROJECT_NAME} PRIVATE ${ASCEND_PATH}/lib64)

# 【关键修复】使用根目录定义的 dynamic_libs，它包含了框架的头文件路径
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${dynamic_libs}
    ${OpenCV_LIBS}
    ascendcl
)

# ==========================================
# 5. 安装规则 (适配根目录的 build.sh)
# ==========================================
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin)