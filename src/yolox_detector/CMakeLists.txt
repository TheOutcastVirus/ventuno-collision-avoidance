cmake_minimum_required(VERSION 3.22)
project(yolox_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ROS 2 ─────────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ── ExecuTorch ────────────────────────────────────────────────────────────────
# Build ExecuTorch from source and point CMAKE_PREFIX_PATH at the install dir,
# or set EXECUTORCH_INSTALL_DIR explicitly:
#   cmake -DEXECUTORCH_INSTALL_DIR=/opt/executorch ..
if(DEFINED EXECUTORCH_INSTALL_DIR)
  list(APPEND CMAKE_PREFIX_PATH ${EXECUTORCH_INSTALL_DIR})
endif()

find_package(executorch REQUIRED)

# ── Backend selection ─────────────────────────────────────────────────────────
option(BUILD_QNN_BACKEND "Build the QTI QNN NPU backend" OFF)

set(SOURCES
  src/main.cpp
  src/yolox_detector_node.cpp
  src/cpu_backend.cpp
  src/postprocess.cpp
)

if(BUILD_QNN_BACKEND)
  list(APPEND SOURCES src/qnn_backend.cpp)
endif()

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(yolox_detector_node ${SOURCES})

target_include_directories(yolox_detector_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(yolox_detector_node PRIVATE
  $<$<BOOL:${BUILD_QNN_BACKEND}>:HAVE_QNN_BACKEND>
)

target_link_libraries(yolox_detector_node
  executorch
  executorch_core
  extension_module_static
  xnnpack_backend
  ${OpenCV_LIBS}
  $<$<BOOL:${BUILD_QNN_BACKEND}>:qnn_executorch_backend>
)

ament_target_dependencies(yolox_detector_node
  rclcpp
  sensor_msgs
  vision_msgs
  image_transport
  cv_bridge
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS yolox_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
