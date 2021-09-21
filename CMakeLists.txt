cmake_minimum_required(VERSION 3.8)

project("Project")

find_package( OpenCV REQUIRED )

add_executable( ${PROJECT_NAME} 
	src/main.cpp
	src/results_writer.cpp
)

find_package(OpenCV REQUIRED)
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries( ${PROJECT_NAME} 
	${OpenCV_LIBS} 
)